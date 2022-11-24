classdef FORWARD_SOLVER_MIE < FORWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        utility_border;
        boundary_thickness_pixel
        ROI;
        Greenp;
        rads;
        psi;
        PSI;
        eye_3;
        
        V;
        initial_ZP_3;
        pole_num;
        
        attenuation_mask;
        pixel_step_size;

        refocusing_util;
    end
    methods
        function get_default_parameters(h)
            get_default_parameters@FORWARD_SOLVER(h);
            %specific parameters
            h.parameters.iterations_number=-1;
            h.parameters.padding_source=0;
            h.parameters.boundary_strength =1;
            h.parameters.boundary_thickness = 0;
            h.parameters.boundary_sharpness = 1;%2;
            h.parameters.verbose = false;
            h.parameters.acyclic = true;
            h.parameters.truncated = false;
            h.parameters.lmax = 10;
            h.parameters.divide_section = 1;
            h.parameters.mu0 = 1;
            h.parameters.mu1 = 1; 
            h.parameters.radius = 2.5;
            h.parameters.n_s = 1.4609;
            h.parameters.n_m = 1.3355;
        end

        function h=FORWARD_SOLVER_MIE(params)
            h@FORWARD_SOLVER(params);
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            
            h.refocusing_util=exp(h.utility.refocusing_kernel.*h.utility.image_space.coor{3});
            if (h.parameters.use_GPU)
                h.refocusing_util=gpuArray(h.refocusing_util);
            end
        end
        function set_RI(h,RI)
            RI=single(RI);%single computation are faster
            set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            
            h.initial_ZP_3=size(h.RI,3);%size before adding boundary
            
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end
        function condition_RI(h)
            eps_imag = 0;
            %add boundary to the RI
            %%% -Z - source (padding) - reflection plane - Sample - transmission plane - Absorption layer %%%
            h.RI = cat(3,h.parameters.RI_bg * ones(size(h.RI,1),size(h.RI,2),1,size(h.RI,4),size(h.RI,5),'single'),h.RI);%add one slice to put the source and one to get the reflection
            h.RI = cat(3,h.RI,h.parameters.RI_bg * ones(size(h.RI,1),size(h.RI,2),1,size(h.RI,4),size(h.RI,5),'single'));%add one slice to retrive the RI
            h.ROI = h.create_boundary_RI(); %-CHANGED
            %update the size in the parameters
            h.parameters.size=size(h.RI);
            h.V = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
            if size(h.V,4)==1
                h.V = h.V - 1i.*eps_imag;
            else
                for j1 = 1:3
                    h.V(:,:,:,j1,j1) = h.V(:,:,:,j1,j1) - 1i.*eps_imag;
                end
            end
            
        end
        function ROI = create_boundary_RI(h) % -CHANGED
            warning('choose a higher size boundary to a size which fft is fast ??');
            warning('allow to chose a threshold for remaining energy');
            
            ZP0 = size(h.RI);
        % Set boundary size & absorptivity %-CHANGED
            if length(h.parameters.boundary_thickness) == 1
                h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)/h.parameters.resolution(3));
            elseif length(h.parameters.boundary_thickness) == 3
                h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)./h.parameters.resolution);
            else
                error('h.boundary_thickness_pixel vector dimension should be 1 or 3.')
            end
        
        % Pad boundary
            if (h.parameters.use_GPU)
                h.RI = gpuArray(h.RI);
                if length(h.boundary_thickness_pixel) == 1
                    h.RI = cat(3,h.RI,h.parameters.RI_bg.*ones(size(h.RI,1),size(h.RI,2),h.boundary_thickness_pixel,'single','gpuArray'));
                else
                    h.RI = cat(1,h.RI,h.parameters.RI_bg.*ones(h.boundary_thickness_pixel(1),size(h.RI,2),size(h.RI,3),size(h.RI,4),size(h.RI,5),'single','gpuArray'));
                    h.RI = cat(2,h.RI,h.parameters.RI_bg.*ones(size(h.RI,1),h.boundary_thickness_pixel(2),size(h.RI,3),size(h.RI,4),size(h.RI,5),'single','gpuArray'));
                    h.RI = cat(3,h.RI,h.parameters.RI_bg.*ones(size(h.RI,1),size(h.RI,2),h.boundary_thickness_pixel(3),size(h.RI,4),size(h.RI,5),'single','gpuArray'));
                end
            end
            ROI = [1 ZP0(1) 1 ZP0(2) h.parameters.padding_source+2 ZP0(3)-1];
            
            V_temp = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
            
            if length(h.boundary_thickness_pixel) == 1 %-CHANGED
                x=(1:size(V_temp,3))-floor(size(V_temp,3)/2);x=circshift(x,-floor(size(V_temp,3)/2));
                x=x/(h.boundary_thickness_pixel/2);
                x=circshift(x,size(V_temp,3)-round(h.boundary_thickness_pixel/2));
                val=x;val(abs(x)>=1)=1;val=abs(val);
                val=1-val;
                val(val>h.parameters.boundary_sharpness)=h.parameters.boundary_sharpness;
                val=val./h.parameters.boundary_sharpness;
                h.attenuation_mask{1}=(1-reshape(val,1,1,[]).*h.parameters.boundary_strength);
            else
                for j1 = 1:3
                    x=(1:size(V_temp,j1))-floor(size(V_temp,j1)/2);x=circshift(x,-floor(size(V_temp,j1)/2));
                    x=x/(h.boundary_thickness_pixel(j1)/2);
                    x=circshift(x,size(V_temp,j1)-round(h.boundary_thickness_pixel(j1)/2));
                    val0=x;val0(abs(x)>=1)=1;val0=abs(val0);
                    val0=1-val0;
                    val0(val0>h.parameters.boundary_sharpness)=h.parameters.boundary_sharpness;
                    val0=val0./h.parameters.boundary_sharpness;
                    if j1 == 1
                        h.attenuation_mask{1}=(1-reshape(val0,[],1,1).*h.parameters.boundary_strength);
                    elseif j1 == 2
                        h.attenuation_mask{2}=(1-reshape(val0,1,[],1).*h.parameters.boundary_strength);
                    else
                        h.attenuation_mask{3}=(1-reshape(val0,1,1,[]).*h.parameters.boundary_strength);
                    end
                end
            end
   
            h.RI = potential2RI(V_temp,h.parameters.wavelength,h.parameters.RI_bg);
            if (h.parameters.use_GPU)
                h.RI=gather(h.RI);
            end
        end
        function init(h)
            eps_imag = 0;
            warning('off','all');
            h.utility_border=DERIVE_OPTICAL_TOOL(h.parameters,h.parameters.use_GPU); % the utility for the space with border
            warning('on','all');
            
            if h.parameters.verbose && h.parameters.iterations_number>0
                warning('Best is to set iterations_number to -n for an automatic choice of this so that reflection to the ordern n-1 are taken in accound (transmission n=1, single reflection n=2, higher n=?)');
            end
            
            if h.parameters.use_GPU
                h.RI=single(gpuArray(h.RI));
            end
            h.pole_num=1;
            if h.parameters.vector_simulation
                h.pole_num=3;
            end
            
            h.rads=...
                (h.utility_border.fourier_space.coor{1}./h.utility_border.k0_nm).*reshape([1 0 0],1,1,1,[])+...
                (h.utility_border.fourier_space.coor{2}./h.utility_border.k0_nm).*reshape([0 1 0],1,1,1,[])+...
                (h.utility_border.fourier_space.coor{3}./h.utility_border.k0_nm).*reshape([0 0 1],1,1,1,[]);
                       
            h.eye_3=reshape(eye(3),1,1,1,3,3);
            if h.parameters.use_GPU
                h.eye_3=gpuArray(h.eye_3);
            end
            if h.parameters.truncated
                S=2.*pi.*sqrt(abs(h.utility_border.fourier_space.coor{1}).^2+abs(h.utility_border.fourier_space.coor{2}).^2+abs(h.utility_border.fourier_space.coor{3}).^2);
                L=norm([h.utility_border.image_space.size{1}.*h.utility_border.image_space.res{1}...
                    h.utility_border.image_space.size{2}.*h.utility_border.image_space.res{2}...
                    h.utility_border.image_space.size{3}.*h.utility_border.image_space.res{3}]);
                K=2.*pi.*h.utility_border.k0_nm;
                K = sqrt(K^2 + 1i.*eps_imag);
                h.Greenp=(1 ./(S.^2-K.^2)...
                    -1./2./S...
                    .*(exp(1i.*(S+K).*L)./(S+K)+exp(1i.*(-S+K).*L)./(S-K))...
                    );  
                h.Greenp(S==0) = -(1-exp(1i*K*L)+1i*K*L*exp(1i*K*L)) / K^2;
                h.Greenp(S==K)= 1/2/K.*...
                    (1i*L + ...
                    (1-exp(1i.*K.*L)) ./ 2 ./ K...
                    );
            else
                h.Greenp = 1 ./ (4*pi^2.*abs(...
                    h.utility_border.fourier_space.coor{1}.^2 + ...
                    h.utility_border.fourier_space.coor{2}.^2 + ...
                    h.utility_border.fourier_space.coor{3}.^2 ...
                    )-(2*pi*h.utility_border.k0_nm)^2-1i*eps_imag);
            end
            
            if h.parameters.use_GPU
                for j1 = 1:length(h.attenuation_mask)
                    h.attenuation_mask{j1} = gpuArray(h.attenuation_mask{j1});
                end
                h.rads = gpuArray(single(h.rads));
                h.Greenp = gpuArray(single(h.Greenp));
            end
            
            h.Greenp=ifftshift(ifftshift(ifftshift(h.Greenp,1),2),3);
            h.rads=ifftshift(ifftshift(ifftshift(h.rads,1),2),3);
            
            if h.parameters.verbose
                figure('units','normalized','outerposition',[0 0 1 1])
                colormap hot
            end
            
            h.RI=gather(h.RI);
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            if ~h.parameters.use_GPU
                input_field=single(input_field);
            else
                h.RI=single(gpuArray(h.RI));
                input_field=single(gpuArray(input_field));
            end
            if size(input_field,3)>1 &&~h.parameters.vector_simulation
                error('the source is 2D but parameter indicate a vectorial simulation');
            elseif size(input_field,3)==1 && h.parameters.vector_simulation
                error('the source is 3D but parameter indicate a non-vectorial simulation');
            end
            if h.parameters.verbose && size(input_field,3)==1
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
            if h.pole_num==3
                out_pol=2;
            end
            fields_trans=[];
            if h.parameters.return_transmission
                fields_trans=ones(size(h.V,1),size(h.V,2),out_pol,size(input_field,4),'single');
                fields_trans=fields_trans(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),:,:,:);
            end
            fields_ref=[];
            if h.parameters.return_reflection
                fields_ref=ones(size(h.V,1),size(h.V,2),out_pol,size(input_field,4),'single');
                fields_ref=fields_ref(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),:,:,:);
            end
            fields_3D=[];
            if h.parameters.return_3D
                fields_3D=ones(size(h.V,1),size(h.V,2),size(h.V,3),size(input_field,3),size(input_field,4),'single');
                fields_3D = fields_3D(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:,:);
            end
            
            for field_num=1:size(input_field,4)
                [field_3D, field_trans, field_ref]=h.solve_raw(input_field(:,:,:,field_num), field_num);
                %crop and remove near field (3D to 2D field)
                
                if h.parameters.return_3D
                    fields_3D(:,:,:,:,field_num)=gather(field_3D);
                end
                if h.parameters.return_transmission
                    fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
                end
                if h.parameters.return_reflection
                    % {
                    fields_ref(:,:,:,field_num)=gather(squeeze(field_ref));
                    %}
                end
            end
            
        end
        function [field_3D, field_trans, field_ref]=solve_raw(h,source,field_num)
            Field = h.V * 0;
            Field = repmat(Field, [1 1 1 3]);
            if field_num == 1
                kx = 0; ky = 0;
            else
                [kx, ky] = find(source == max(source(:)));
                kx = kx - floor(size(source,1)/2) - 1;
                ky = ky - floor(size(source,2)/2) - 1;
                kx = kx * utility.fourier_space.res{1};
                ky = ky * utility.fourier_space.res{2};
            end
            source = reshape(source, [size(source,1),size(source,2),1,size(source,3)]).*...
                exp(h.utility_border.refocusing_kernel.*h.parameters.resolution(3).*reshape((-floor(h.initial_ZP_3/2)-1)+((1-1-h.parameters.padding_source):1:(h.initial_ZP_3+1)),[1 1 h.initial_ZP_3+2+h.parameters.padding_source]));
            source = fftshift(ifft2(ifftshift(source)));
            incident_field = source;
            
            
            [h.parameters.n, h.parameters.m]=makelmidx(h.parameters.lmax); 
            h.parameters.alm=zeros(h.parameters.lmax,2*h.parameters.lmax+1); 
            h.parameters.blm=h.parameters.alm;
            
            % Obtain T-matrix - The first T is scattered mode, 2nd T is the internal mode.
            k_m = h.utility_border.k0_nm * 2 * pi;
            k_s = k_m * h.parameters.n_s / h.parameters.RI_bg;
            
            [h.parameters.T_ext,h.parameters.T_int,h.parameters.A,h.parameters.B,h.parameters.C,h.parameters.D] =...
                tmatrix_mie_v2(h.parameters.lmax,k_m,k_s,h.parameters.radius,h.parameters.mu0,h.parameters.mu1);
            [xf0, yf0,zf0] = ndgrid(gather(h.utility_border.image_space.coor{1}), gather(h.utility_border.image_space.coor{2}),gather(h.utility_border.image_space.coor{3})); clear nx ny nz
            [theta, phi, rad] = xcart2sph(xf0, yf0, zf0);  % Spherical grids
            rad = double(rad(:));phi = double(phi(:));theta = double(theta(:));
            theta(isnan(theta))=0;
            rho = k_m *rad; % [kr] unitless radial variable
            rho_s=k_s *rad; % [kr] unitless radial variable
            % Spherical unit vectors into cartesian vectors
            Rx = getTransformationMatrix(gather(phi(:)), gather(theta(:)));

            % k vector & deflected polarization
            kvecs = [kx ky];
            kvecs(3) = sqrt(k_m^2 - norm(kvecs)^2);
            [ktheta,kphi,~] = xcart2sph(kvecs(:,1),kvecs(:,2),kvecs(:,3));
            pol_new = parallel_transport_pol(kvecs); % Choose s-pol only   
            clear dy1 dy2 eps_regul kx ky kx kz1 kz2   
            % Input field decomposition 
            R = permute(getTransformationMatrix(kphi, ktheta),[2,1,3]); % Transpose = Inverse : from cartesian to spherical coordinate
            pol_new_sph = gather(squeeze(sum(R .* reshape(transpose(pol_new), [1 3 size(pol_new,1)]),2)));
            
            %% Start computation
            l=1;
            while true
            % Initialize parameters
                cl=sqrt((2*l+1)/4/pi/l/(l+1));
        %         [B,C,~] = basic_wavevectors(l,ktheta);
                [BB,CC,~] = vsh_MS(l,ktheta);
                BB(isnan(BB)) = 0; CC(isnan(CC)) = 0;
                ms = reshape(-l:1:l,[2*l+1 1 1]);
                h.parameters.alm(l,1:size(BB,1)) = transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^l .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])).* conj(CC) .* reshape(transpose(pol_new_sph),[1 size(pol_new_sph,2),3]),2),3));
                h.parameters.blm(l,1:size(BB,1)) =  transpose(sum(sum(4.* pi.* (-1).^ms .* (1i).^(l-1) .* cl .* exp(-1i.*ms.*reshape(kphi,[1 length(kphi) 1])) .* conj(BB) .* reshape(transpose(pol_new_sph),[1 size(pol_new_sph,2),3]),2),3));
                if l == h.parameters.lmax
                    break;
                end
                l=l+1;
            end

            lmax=l;
            h.parameters.alm=h.parameters.alm(1:lmax,1:2*lmax+1);h.parameters.blm=h.parameters.blm(1:lmax,1:2*lmax+1);
            %   imagesc(abs(alm_tomotrap))
        % T-matrix method
            % Coefficient normalization
        %     normalization_power=sqrt(sum(abs(alm(:)).^2+abs(blm(:)).^2));
            normalization_power=1;
            a0s=h.parameters.alm./normalization_power;
            b0s = h.parameters.blm./normalization_power;
            as=(zeros((lmax+1)^2-1,1));bs=(zeros((lmax+1)^2-1,1));
            for i2=1:lmax
                as(i2^2:(i2+1)^2-1)=a0s(i2,1:(i2+1)^2-i2^2);
                bs(i2^2:(i2+1)^2-1)=b0s(i2,1:(i2+1)^2-i2^2);
            end
            as=sparse(as);bs=sparse(bs);
        %  figure(),subplot(211),plot(abs(A)), hold on;plot(abs(B)), hold on;plot(abs(C)), hold on;plot(abs(D)), hold on;legend('A','B','C','D'),subplot(212),plot(angle(A)), hold on;plot(angle(B)), hold on;plot(angle(C)), hold on;plot(angle(D)), hold on;legend('A','B','C','D')

        % Output mode coefficients
            a2s = as;b2s = bs;
            pq = h.parameters.T_ext * [ a2s; b2s ];   p_out = pq(1:length(pq)/2);q_out = pq(length(pq)/2+1:end);
            pq = h.parameters.T_int *[ a2s; b2s ];   p_in = pq(1:length(pq)/2);q_in = pq(length(pq)/2+1:end);

            E_scat_T=zeros(numel(phi),3);%E0_T=E_scat_T; 

            % Make scattered field
            in_flag = rho < k_m*h.parameters.radius; in_flag=in_flag(:);out_flag = rho >= k_m*h.parameters.radius; out_flag=out_flag(:);
            M_in = E_scat_T; N_in = E_scat_T; M_outj = E_scat_T; N_outj = E_scat_T; M_outh = E_scat_T; N_outh = E_scat_T;  
%             E0_T = E_scat_T;
            %%
            tic;
            % Memory allocation
            for bulk = 1: h.parameters.divide_section
                length_bulk = ceil(length(theta) / h.parameters.divide_section);
                jj = (1+length_bulk * (bulk-1)) : min(length(theta), bulk*length_bulk);
                for l = 1:lmax
                    %%
                    [BB,CC,P] = vsh_MS(l,theta(jj));
                    BB(isnan(BB)) = 0; CC(isnan(CC)) = 0; P(isnan(P)) = 0;
                    %%
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
                        disp(['Volume section: ' num2str(bulk) ' / ' num2str(h.parameters.divide_section)])
                        disp(['l: ' num2str(l) ' / ' num2str(lmax)])
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
            out_flag = squeeze(reshape(out_flag,[size(xf0)]));
            E_scat_T = transpose(squeeze(sum(Rx .* reshape(transpose(E_scat_T), [1 3 size(E_scat_T,1)]),2)));
            E_scat_T = squeeze(reshape(E_scat_T,[size(xf0), 3]));
        
            
            size(out_flag)
            size(Field)
            size(E_scat_T)
           
            
            Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.parameters.padding_source):(h.ROI(6)+1),:,:) = ...
                incident_field.*out_flag(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.parameters.padding_source):(h.ROI(6)+1),:,:) +...
                E_scat_T(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.parameters.padding_source):(h.ROI(6)+1),:,:);

            if h.parameters.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, colorbar, axis off,drawnow
                colormap hot
            end
            
            
            
            % Retrieve final result
            field_3D = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:);
            
            field_trans = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4),h.ROI(6)+1,:);
            field_trans=squeeze(field_trans);
            field_trans=fftshift(fft2(ifftshift(field_trans)));
            [field_trans] = h.transform_field_2D(field_trans);
            field_trans=field_trans.*exp(h.utility_border.refocusing_kernel.*h.parameters.resolution(3).*(floor(h.initial_ZP_3/2)+1-(h.initial_ZP_3+1)));
            field_trans=field_trans.*h.utility_border.NA_circle;%crop to the objective NA
            field_trans=fftshift(ifft2(ifftshift(field_trans)));

            Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.parameters.padding_source):(h.ROI(6)+1),:,:) = ...
                gather(single(Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)-1-h.parameters.padding_source):(h.ROI(6)+1),:,:) - incident_field));
            field_ref = Field(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4),h.ROI(5)-1,:);
            field_ref=squeeze(field_ref);
            field_ref=fftshift(fft2(ifftshift(field_ref)));
            [field_ref] = h.transform_field_2D_reflection(field_ref);
            field_ref=field_ref.*exp(h.utility_border.refocusing_kernel.*h.parameters.resolution(3).*(-floor(h.initial_ZP_3/2)-1));
            field_ref=field_ref.*h.utility_border.NA_circle;%crop to the objective NA
            field_ref=fftshift(ifft2(ifftshift(field_ref)));
            
        end
        
    end
end


