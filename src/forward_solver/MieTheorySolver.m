classdef MieTheorySolver < ForwardSolver
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
    end
    methods
        function obj=MieTheorySolver(params)
            obj@ForwardSolver(params);
        end
        function set_RI(obj, RI)
            obj.RI=single(RI);%single computation are faster
            obj.initial_Nsize=size(obj.RI);%size before adding boundary
            obj.condition_RI();%modify the RI (add padding and boundary)
            obj.init();%init the parameter for the forward model
        end
        function condition_RI(obj)
            %add boundary to the RI
            %%% -Z - source (padding) - reflection plane - Sample - transmission plane - Absorption layer %%%
            obj.RI = padarray(obj.RI,[0 0 1],obj.RI_bg,'both');
            obj.create_boundary_RI();
        end
        function create_boundary_RI(obj)
            Nsize = size(obj.RI);
            % Set boundary size & absorptivity
            if length(obj.boundary_thickness) == 1
                obj.boundary_thickness_pixel = [0 0 round((obj.boundary_thickness*obj.wavelength/obj.RI_bg)/obj.resolution(3))];
            elseif length(obj.boundary_thickness) == 3
                obj.boundary_thickness_pixel = round((obj.boundary_thickness*obj.wavelength/obj.RI_bg)./obj.resolution);
            else
                error('obj.boundary_thickness_pixel vector dimension should be 1 or 3.')
            end
        
            % Pad boundary;
            if (obj.use_GPU)
                obj.RI = padarray(obj.RI, obj.boundary_thickness_pixel, obj.RI_bg, 'post');
            end
            obj.ROI = [1 Nsize(1) 1 Nsize(2) obj.padding_source+2 Nsize(3)-1];
        end
        function init(obj)
            Nsize=size(obj.RI);
            warning('off','all');
            obj.utility=derive_utility(obj, Nsize); % the utility for the space with border
            warning('on','all');
        end
        function [Efield]=solve(obj,input_field)
            if ~obj.use_GPU
                input_field=single(input_field);
            else
                obj.RI=single(gpuArray(obj.RI));
                input_field=single(gpuArray(input_field));
            end

            assert(size(input_field,3) == 2, 'The 3rd dimension of input_field should indicate polarization')
            if obj.verbose && size(input_field,3)==1
                warning('Input is scalar but scalar equation is less precise');
            end
            if size(input_field,3)>2
                error('Input field must be either a scalar or a 2D vector');
            end
            
            input_field=fftshift(fft2(ifftshift(input_field)));
            %2D to 3D field
            source = obj.transform_field_3D(input_field);
            Field = complex(zeros([size(obj.RI,1:3) 3], 'single'));
            % defined a k-vector for the illuminated plane & wavenumbers in both sample and background medium
            [kx, ky] = find(source == max(source(:)));
            kx = kx - floor(size(source,1)/2) - 1;
            ky = ky - floor(size(source,2)/2) - 1;
            kx = kx * obj.utility.fourier_space.res{1};
            ky = ky * obj.utility.fourier_space.res{2};

            k_m = obj.utility.k0_nm * 2 * pi;
            k_s = k_m * obj.n_s / obj.RI_bg;
            k_vector = [kx ky sqrt(k_m^2 - kx^2 - ky^2)]; %%% sqrt(k_m^2 - norm(k_vector)^2); ?? why norm?
            [ktheta,kphi,~] = xcart2sph(k_vector(1),k_vector(2),k_vector(3));
            % generate volumetric source
            source = reshape(source, size(source, 1), size(source, 2), 1, size(source, 3)) .* ...
                exp(obj.utility.refocusing_kernel.*obj.resolution(3) .* reshape(-floor(obj.initial_Nsize(3)/2)-obj.padding_source-1:ceil(obj.initial_Nsize(3)/2),1, 1, []));
            source = fftshift(ifft2(ifftshift(source)));
            incident_field = source;
            % Obtain T-matrix - The first T is scattered mode, 2nd T is the internal mode.
            [T_ext, T_int] = tmatrix_mie_v2(obj.lmax,k_m,k_s,obj.radius,obj.mu0,obj.mu1);
            [xf0, yf0, zf0] = ndgrid(gather(obj.utility.image_space.coor{1}), gather(obj.utility.image_space.coor{2}),gather(obj.utility.image_space.coor{3}));
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
            alm = zeros(obj.lmax,2*obj.lmax+1); 
            blm = zeros('like', alm);
            for l = 1:obj.lmax
            % Initialize parameters
                cl=sqrt((2*l+1)/4/pi/l/(l+1));
        %         [B,C,~] = basic_wavevectors(l,ktheta);
                [BB, CC] = vsh_MS(l,ktheta);
                BB(isnan(BB)) = 0; CC(isnan(CC)) = 0;
                ms = reshape(-l:1:l,[2*l+1 1 1]);
                alm(l,1:size(BB,1)) = transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^l .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])).* conj(CC) .* reshape(transpose(pol_new_sph), 1, [], 3),2),3));
                blm(l,1:size(BB,1)) =  transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^(l-1) .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])) .* conj(BB) .* reshape(transpose(pol_new_sph), 1, [], 3),2),3));
            end
            % T-matrix method
            % Coefficient normalization
            normalization_power=1;
            a0s=alm/normalization_power;
            b0s = blm/normalization_power;
            a2s=zeros((obj.lmax+1)^2-1,1);
            b2s=zeros((obj.lmax+1)^2-1,1);
            for l_val=1:obj.lmax
                a2s(l_val^2:(l_val+1)^2-1)=a0s(l_val,1:(l_val+1)^2-l_val^2);
                b2s(l_val^2:(l_val+1)^2-1)=b0s(l_val,1:(l_val+1)^2-l_val^2);
            end
            a2s=sparse(a2s);b2s=sparse(b2s);
            % Output mode coefficients
            pq = T_ext * [ a2s; b2s ];
            p_out = pq(1:length(pq)/2);
            q_out = pq(length(pq)/2+1:end);
            pq = T_int * [ a2s; b2s ];
            p_in = pq(1:length(pq)/2);
            q_in = pq(length(pq)/2+1:end);
            E_scat_T=zeros(numel(phi),3);
            % Make scattered field
            in_flag = rho < k_m*obj.radius; in_flag=in_flag(:);out_flag = rho >= k_m*obj.radius; out_flag=out_flag(:);
            % main
            tic;
            if kx == 0 && ky == 0
                m_list = [-1 1];
            else
                m_list = -l:1:l;
            end
            for bulk = 1: obj.divide_section
                length_bulk = ceil(length(theta) / obj.divide_section);
                jj = (1+length_bulk * (bulk-1)) : min(length(theta), bulk*length_bulk);
                M_in = zeros(numel(jj),3);
                N_in = zeros(numel(jj),3);
                M_outh = zeros(numel(jj),3);
                N_outh = zeros(numel(jj),3);
                for l = 1:obj.lmax
                    [BB,CC,P] = vsh_MS(l,theta(jj));
                    BB(isnan(BB)) = 0; CC(isnan(CC)) = 0; P(isnan(P)) = 0;
                    cl=sqrt((2*l+1)/4/pi/l/(l+1));
                    % Spherical Bessel functions for internal field
                    j_s = sbesselj(l, rho_s(jj));j_s=j_s(:);j_s(isnan(j_s))=0;
                    dxi_s = ricbesjd(l, rho_s(jj));	dxi_s=dxi_s(:); dxi_s(isnan(dxi_s))=0;
                    j_rho_s = H_Rho(j_s,rho_s(jj),l); j_rho_s(isnan(j_rho_s))=0;
                    dxi_rho_s = Dxi_Rho(dxi_s,rho_s(jj),l); dxi_rho_s(isnan(dxi_rho_s))=0;
                    % Spherical Bessel functions for external field
                    h_m = sbesselh1(l, rho(jj));h_m=h_m(:);h_m(isnan(h_m))=0;
                    dxih_m = ricbesh1d(l, rho(jj));dxih_m=dxih_m(:); dxih_m(isnan(dxih_m))=0;
                    h_rho_m = H_Rho(h_m,rho(jj),l); h_rho_m(isnan(h_rho_m))=0;
                    dxih_rho_m = Dxi_Rho(dxih_m,rho(jj),l); dxih_rho_m(isnan(dxih_rho_m))=0;
                    for m = m_list
                        phase = exp(1i.*m.*phi(jj));
                        clc;
                        disp(['Volume section: ' num2str(bulk) ' / ' num2str(obj.divide_section)])
                        disp(['l: ' num2str(l) ' / ' num2str(obj.lmax)])
                        disp(['m: ' num2str(m)])
                        idx = m + l + 1;
                        M_in(:)   = j_s .* reshape(CC(idx,:,:),[],3);
                        N_in(:)   = l*(l+1).*j_rho_s.*reshape(P(idx,:,:),[],3) + dxi_rho_s .* reshape(BB(idx,:,:),[],3);

                        M_outh(:) = h_m .* reshape(CC(idx,:,:),[],3);
                        N_outh(:) = l*(l+1).*h_rho_m.*reshape(P(idx,:,:),[],3) + dxih_rho_m .* reshape(BB(idx,:,:),[],3);

                        E_scat_T(jj,:) = E_scat_T(jj,:) + (out_flag(jj) .* (full(p_out(l*(l+1)+m)) .* M_outh + full(q_out(l*(l+1)+m)) .* N_outh) ... 
                                                        + in_flag(jj) .* (full(p_in(l*(l+1)+m)) .* M_in + full(q_in(l*(l+1)+m)) .* N_in)) .* ((-1)^m * cl *phase);
                    end
                end
            end
            toc;
            out_flag = squeeze(reshape(out_flag,size(xf0)));
            E_scat_T = transpose(squeeze(sum(Rx .* reshape(transpose(E_scat_T), [1 3 size(E_scat_T,1)]),2)));
            E_scat_T = squeeze(reshape(E_scat_T,[size(xf0), 3]));
        
            Field(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),(obj.ROI(5)-1-obj.padding_source):(obj.ROI(6)+1),:,:) = ...
                incident_field.*out_flag(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),(obj.ROI(5)-1-obj.padding_source):(obj.ROI(6)+1),:,:) +...
                E_scat_T(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),(obj.ROI(5)-1-obj.padding_source):(obj.ROI(6)+1),:,:);

            if obj.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))')));axis image; colorbar; axis off;drawnow
                colormap hot
            end

            % Retrieve final result
            Efield = gather(Field(obj.ROI(1):obj.ROI(2), obj.ROI(3):obj.ROI(4), obj.ROI(5):obj.ROI(6),:));    
        end
    end
end


