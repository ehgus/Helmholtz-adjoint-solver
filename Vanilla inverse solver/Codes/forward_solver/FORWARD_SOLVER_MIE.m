classdef FORWARD_SOLVER_MIE < FORWARD_SOLVER
    properties
        % scattering object w/ boundary
        radius = 2.5;
        n_s = 1.4609;
        boundary_thickness_pixel;
        padding_source = 0;
        boundary_thickness = 0; 
        ROI;
        initial_Nsize;
        % Mie scattering option
        lmax = 10;
        mu0 = 1;
        mu1 = 1;
        % acceleration
        divide_section = 1;
        % return values
        return_3D = true;
        return_transmission = false;
        return_reflection = false;
    end
    methods
        function h=FORWARD_SOLVER_MIE(params)
            h@FORWARD_SOLVER(params);
        end
        function set_RI(h,RI)
            h.RI=single(RI);%single computation are faster
            h.initial_Nsize=size(h.RI);%size before adding boundary
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end
        function condition_RI(h)
            %add boundary to the RI
            %%% -Z - source (padding) - reflection plane - Sample - transmission plane - Absorption layer %%%
            h.RI = padarray(h.RI,[0 0 1],h.RI_bg,'both');
            h.create_boundary_RI();
        end
        function create_boundary_RI(h)
            warning('allow to chose a threshold for remaining energy');
            Nsize = size(h.RI);
            % Set boundary size & absorptivity
            if length(h.boundary_thickness) == 1
                h.boundary_thickness_pixel = [0 0 round((h.boundary_thickness*h.wavelength/h.RI_bg)/h.resolution(3))];
            elseif length(h.boundary_thickness) == 3
                h.boundary_thickness_pixel = round((h.boundary_thickness*h.wavelength/h.RI_bg)./h.resolution);
            else
                error('h.boundary_thickness_pixel vector dimension should be 1 or 3.')
            end
        
            % Pad boundary;
            if (h.use_GPU)
                h.RI = padarray(h.RI, h.boundary_thickness_pixel, h.RI_bg, 'post');
            end
            h.ROI = [1 Nsize(1) 1 Nsize(2) h.padding_source+2 Nsize(3)-1];
        end
        function init(h)
            Nsize=size(h.RI);
            warning('off','all');
            h.utility=derive_utility(h, Nsize); % the utility for the space with border
            warning('on','all');
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            if ~h.use_GPU
                input_field=single(input_field);
            else
                h.RI=single(gpuArray(h.RI));
                input_field=single(gpuArray(input_field));
            end
            if size(input_field,3)>1 &&~h.vector_simulation
                error('the source is 2D but parameter indicate a vectorial simulation');
            elseif size(input_field,3)==1 && h.vector_simulation
                error('the source is 3D but parameter indicate a non-vectorial simulation');
            end
            if h.verbose && size(input_field,3)==1
                warning('Input is scalar but scalar equation is less precise');
            end
            if size(input_field,3)>2
                error('Input field must be either a scalar or a 2D vector');
            end
            
            input_field=fftshift(fft2(ifftshift(input_field)));
            %2D to 3D field
            [input_field] = h.transform_field_3D(input_field);
            %compute
            out_pol=1;
            if h.vector_simulation
                out_pol=2;
            end
            fields_trans=[];
            if h.return_transmission
                fields_trans=ones(h.ROI(2)-h.ROI(1)+1,h.ROI(4)-h.ROI(3)+1,out_pol,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.return_reflection
                fields_ref=ones(h.ROI(2)-h.ROI(1)+1,h.ROI(4)-h.ROI(3)+1,out_pol,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.return_3D
                fields_3D=ones(h.ROI(2)-h.ROI(1)+1, h.ROI(4)-h.ROI(3)+1, h.ROI(6)-h.ROI(5)+1, size(input_field,3), size(input_field,4), 'single');
            end
            for field_num=1:size(input_field,4)
                [field_3D, field_trans, field_ref]=h.solve_forward(input_field(:,:,:,field_num), field_num);
                %crop and remove near field (3D to 2D field)
                if h.return_3D
                    fields_3D(:,:,:,:,field_num)=gather(field_3D);
                end
                if h.return_transmission
                    fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
                end
                if h.return_reflection
                    fields_ref(:,:,:,field_num)=gather(squeeze(field_ref));
                end
            end
            
        end
        function [field_3D, field_trans, field_ref]=solve_forward(h,source,field_num)
            Field = complex(zeros([size(h.RI,1:3) 3], 'single'));
            % defined a k-vector for the illuminated plane & wavenumbers in both sample and background medium
            if field_num == 1
                kx = 0; ky = 0;
            else
                [kx, ky] = find(source == max(source(:)));
                kx = kx - floor(size(source,1)/2) - 1;
                ky = ky - floor(size(source,2)/2) - 1;
                kx = kx * utility.fourier_space.res{1};
                ky = ky * utility.fourier_space.res{2};
            end
            k_m = h.utility.k0_nm * 2 * pi;
            k_s = k_m * h.n_s / h.RI_bg;
            k_vector = [kx ky sqrt(k_m^2 - kx^2 - ky^2)]; %%% sqrt(k_m^2 - norm(k_vector)^2); ?? why norm?
            [ktheta,kphi,~] = xcart2sph(k_vector(1),k_vector(2),k_vector(3));
            % generate volumetric source
            source = source .* exp(h.utility.refocusing_kernel.*h.resolution(3) .* reshape((-floor(h.initial_Nsize(3)/2)-1)+((1-1-h.padding_source):1:(h.initial_Nsize(3)+1)),1, 1, []));
            source = fftshift(ifft2(ifftshift(source)));
            incident_field = source;
            % Obtain T-matrix - The first T is scattered mode, 2nd T is the internal mode.
            parameters = struct();
            [parameters.T_ext, parameters.T_int] = tmatrix_mie_v2(h.lmax,k_m,k_s,h.radius,h.mu0,h.mu1);
            [xf0, yf0, zf0] = ndgrid(gather(h.utility.image_space.coor{1}), gather(h.utility.image_space.coor{2}),gather(h.utility.image_space.coor{3}));
            [theta, phi, rad] = xcart2sph(xf0, yf0, zf0);  % Spherical grids
            rad = double(rad(:));phi = double(phi(:));theta = double(theta(:));
            theta(isnan(theta))=0;
            rho = k_m * rad;    % [kr] unitless radial variable
            rho_s = k_s * rad;  % [kr] unitless radial variable
            % Spherical unit vectors into cartesian vectors
            Rx = getTransformationMatrix(gather(phi(:)), gather(theta(:)));

            % deflected polarization
            pol_new = parallel_transport_pol(k_vector); % Choose s-pol only   2   
            % Input field decomposition 
            R = permute(getTransformationMatrix(kphi, ktheta),[2,1,3]); % Transpose = Inverse : from cartesian to spherical coordinate
            pol_new_sph = gather(squeeze(sum(R .* reshape(transpose(pol_new), [1 3 size(pol_new,1)]),2)));
            
            %% Start computation
            parameters.alm = zeros(h.lmax,2*h.lmax+1); 
            parameters.blm = zeros('like', parameters.alm);
            for l = 1:h.lmax
            % Initialize parameters
                cl=sqrt((2*l+1)/4/pi/l/(l+1));
        %         [B,C,~] = basic_wavevectors(l,ktheta);
                [BB, CC] = vsh_MS(l,ktheta);
                BB(isnan(BB)) = 0; CC(isnan(CC)) = 0;
                ms = reshape(-l:1:l,[2*l+1 1 1]);
                parameters.alm(l,1:size(BB,1)) = transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^l .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])).* conj(CC) .* reshape(transpose(pol_new_sph), 1, [], 3),2),3));
                parameters.blm(l,1:size(BB,1)) =  transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^(l-1) .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])) .* conj(BB) .* reshape(transpose(pol_new_sph), 1, [], 3),2),3));
            end
            % T-matrix method
            % Coefficient normalization
            normalization_power=1;
            a0s=parameters.alm/normalization_power;
            b0s = parameters.blm/normalization_power;
            a2s=zeros((h.lmax+1)^2-1,1);
            b2s=zeros((h.lmax+1)^2-1,1);
            for i2=1:h.lmax
                a2s(i2^2:(i2+1)^2-1)=a0s(i2,1:(i2+1)^2-i2^2);
                b2s(i2^2:(i2+1)^2-1)=b0s(i2,1:(i2+1)^2-i2^2);
            end
            a2s=sparse(a2s);b2s=sparse(b2s);
            % Output mode coefficients
            pq = parameters.T_ext * [ a2s; b2s ];   p_out = pq(1:length(pq)/2);q_out = pq(length(pq)/2+1:end);
            pq = parameters.T_int *[ a2s; b2s ];   p_in = pq(1:length(pq)/2);q_in = pq(length(pq)/2+1:end);
            E_scat_T=zeros(numel(phi),3);
            % Make scattered field
            in_flag = rho < k_m*h.radius; in_flag=in_flag(:);out_flag = rho >= k_m*h.radius; out_flag=out_flag(:);
            M_in = E_scat_T; N_in = E_scat_T; M_outj = E_scat_T; N_outj = E_scat_T; M_outh = E_scat_T; N_outh = E_scat_T;  
            % main
            tic;
            for bulk = 1: h.divide_section
                length_bulk = ceil(length(theta) / h.divide_section);
                jj = (1+length_bulk * (bulk-1)) : min(length(theta), bulk*length_bulk);
                for l = 1:h.lmax
                    [BB,CC,P] = vsh_MS(l,theta(jj));
                    BB(isnan(BB)) = 0; CC(isnan(CC)) = 0; P(isnan(P)) = 0;
                    cl=sqrt((2*l+1)/4/pi/l/(l+1));
                    % Spherical Bessel functions for internal field
                    j_s = sbesselj(l, rho_s(jj));j_s=j_s(:);j_s(isnan(j_s))=0;
                    dxi_s = ricbesjd(l, rho_s(jj));	dxi_s=dxi_s(:); dxi_s(isnan(dxi_s))=0;
                    j_rho_s = H_Rho(j_s,rho_s(jj),l); j_rho_s(isnan(j_rho_s))=0;
                    dxi_rho_s = Dxi_Rho(dxi_s,rho_s(jj),l); dxi_rho_s(isnan(dxi_rho_s))=0;
                    % Spherical Bessel functions for external field
                    j_m = sbesselj(l, rho(jj));j_m=j_m(:);j_m(isnan(j_m))=0;
                    dxi_m = ricbesjd(l, rho(jj));dxi_m=dxi_m(:); dxi_m(isnan(dxi_m))=0;
                    j_rho_m = H_Rho(j_m,rho(jj),l); j_rho_m(isnan(j_rho_m))=0;
                    dxi_rho_m = Dxi_Rho(dxi_m,rho(jj),l); dxi_rho_m(isnan(dxi_rho_m))=0;

                    h_m = sbesselh1(l, rho(jj));h_m=h_m(:);h_m(isnan(h_m))=0;
                    dxih_m = ricbesh1d(l, rho(jj));dxih_m=dxih_m(:); dxih_m(isnan(dxih_m))=0;
                    h_rho_m = H_Rho(h_m,rho(jj),l); h_rho_m(isnan(h_rho_m))=0;
                    dxih_rho_m = Dxi_Rho(dxih_m,rho(jj),l); dxih_rho_m(isnan(dxih_rho_m))=0;
                    if field_num == 1
                        m_list = [-1 1];
                    else
                        m_list = -l:1:l;
                    end
                    for m = m_list
                        clc,
                        disp(['Volume section: ' num2str(bulk) ' / ' num2str(h.divide_section)])
                        disp(['l: ' num2str(l) ' / ' num2str(h.lmax)])
                        disp(['m: ' num2str(m)])
                        idx = m + l + 1;
                        M_in(jj,:)   = (-1).^m .* cl .* (j_s .* reshape(CC(idx,:,:),[length(phi(jj)),3])) .* exp(1i.*m.*phi(jj));
                        N_in(jj,:)   = (-1).^m .* cl .* (l*(l+1).*j_rho_s.*(reshape(P(idx,:,:),[length(phi(jj)),3])) +...
                            dxi_rho_s .* (reshape(BB(idx,:,:),[length(phi(jj)),3]))) .* exp(1i.*m.*phi(jj));

                        M_outj(jj,:) = (-1).^m .* cl .* (j_m .* reshape(CC(idx,:,:),[length(phi(jj)),3])) .* exp(1i.*m.*phi(jj));
                        N_outj(jj,:) = (-1).^m .* cl .* (l*(l+1).*j_rho_m.*(reshape(P(idx,:,:),[length(phi(jj)),3])) +...
                            dxi_rho_m .* (reshape(BB(idx,:,:),[length(phi(jj)),3]))) .* exp(1i.*m.*phi(jj));

                        M_outh(jj,:) = (-1).^m .* cl .* (h_m .* reshape(CC(idx,:,:),[length(phi(jj)),3])) .* exp(1i.*m.*phi(jj));
                        N_outh(jj,:) = (-1).^m .* cl .* (l*(l+1).*h_rho_m.*(reshape(P(idx,:,:),[length(phi(jj)),3])) +...
                            dxih_rho_m .* (reshape(BB(idx,:,:),[length(phi(jj)),3]))) .* exp(1i.*m.*phi(jj));

                            E_scat_T(jj,:) = (E_scat_T(jj,:) + out_flag(jj) .* ((full(p_out(l*(l+1)+m))) .* M_outh(jj,:)) +...
                                                in_flag(jj) .* single(full(p_in(l*(l+1)+m))) .* M_in(jj,:) +...
                                                              out_flag(jj) .* ((full(q_out(l*(l+1)+m))) .* N_outh(jj,:)) +... 
                                                in_flag(jj) .* single(full(q_in(l*(l+1)+m))) .* N_in(jj,:));
                    end
                end
            end
            toc;
            out_flag = squeeze(reshape(out_flag,size(xf0)));
            E_scat_T = transpose(squeeze(sum(Rx .* reshape(transpose(E_scat_T), [1 3 size(E_scat_T,1)]),2)));
            E_scat_T = squeeze(reshape(E_scat_T,[size(xf0), 3]));
        
            Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.padding_source):(h.ROI(6)+1),:,:) = ...
                incident_field.*out_flag(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.padding_source):(h.ROI(6)+1),:,:) +...
                E_scat_T(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.padding_source):(h.ROI(6)+1),:,:);

            if h.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))')));axis image; colorbar; axis off;drawnow
                colormap hot
            end

            % Retrieve final result
            field_3D = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:);
            
            field_trans = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4),h.ROI(6)+1,:);
            field_trans=squeeze(field_trans);
            field_trans=fftshift(fft2(ifftshift(field_trans)));
            [field_trans] = h.transform_field_2D(field_trans);
            field_trans=field_trans.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(floor(h.initial_Nsize(3)/2)+1-(h.initial_Nsize(3)+1)));
            field_trans=field_trans.*h.utility.NA_circle;%crop to the objective NA
            field_trans=fftshift(ifft2(ifftshift(field_trans)));

            Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.padding_source):(h.ROI(6)+1),:,:) = ...
                gather(single(Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.padding_source):(h.ROI(6)+1),:,:) - incident_field));
            field_ref = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4),h.ROI(5)-1,:);
            field_ref=squeeze(field_ref);
            field_ref=fftshift(fft2(ifftshift(field_ref)));
            [field_ref] = h.transform_field_2D_reflection(field_ref);
            field_ref=field_ref.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(-floor(h.initial_Nsize(3)/2)-1));
            field_ref=field_ref.*h.utility.NA_circle;%crop to the objective NA
            field_ref=fftshift(ifft2(ifftshift(field_ref)));
            
        end
        
    end
end


