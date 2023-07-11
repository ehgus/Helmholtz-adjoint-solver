classdef MieTheorySolver < ForwardSolver
    properties
        % scattering object w/ boundary
        radius = 2.5;
        RI_sp = 1.4609;
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
            obj.utility=derive_utility(obj, size(obj.RI)); % the utility for the space with border
        end
        function [Efield]=solve(obj,current_source)
            assert(isa(current_source,'PlaneSource'), "PlaneSource is only supported")
            assert(current_source.direction == 3, "Light should propagate along z-axis")
            % defined a k-vector for the illuminated plane & wavenumbers in both sample and background medium
            kx = current_source.horizontal_k_vector(1);
            ky = current_source.horizontal_k_vector(2);

            k_m = 2 * pi * obj.RI_bg / obj.wavelength;
            k_s = 2 * pi * obj.RI_sp / obj.wavelength;
            k_vector = [kx ky sqrt(k_m^2 - kx^2 - ky^2)];
            [ktheta,kphi,~] = xcart2sph(k_vector(1),k_vector(2),k_vector(3));
            % Obtain T-matrix - The first T is scattered mode, 2nd T is the internal mode.
            [T_ext, T_int] = tmatrix_mie_v2(obj.lmax,k_m,k_s,obj.radius,obj.mu0,obj.mu1);
            [xf0, yf0, zf0] = ndgrid(gather(obj.utility.image_space.coor{1}), gather(obj.utility.image_space.coor{2}),gather(obj.utility.image_space.coor{3}));
            [theta, phi, rad] = xcart2sph(xf0, yf0, zf0);  % Spherical grids
            rad = double(rad(:));phi = double(phi(:));theta = double(theta(:));
            rho = k_m * rad;    % [kr] unitless radial variable
            rho_s = k_s * rad;  % [kr] unitless radial variable
            % Spherical unit vectors into cartesian vectors
            Rx = permute(getTransformationMatrix(gather(phi(:)), gather(theta(:))),[3,1,2]);
            % deflected polarization
            pol_new = parallel_transport_pol(k_vector); % Choose s-pol only 
            % Input field decomposition 
            R = permute(getTransformationMatrix(kphi, ktheta),[3,2,1]); % Transpose = Inverse : from cartesian to spherical coordinate
            pol_new_sph = sum(reshape(pol_new,[],1,3) .* reshape(R,[],3,3),3);
            
            %% Start computation
            alm = zeros(obj.lmax,2*obj.lmax+1); 
            blm = zeros('like', alm);
            for l = 1:obj.lmax
            % Initialize parameters
                cl=sqrt((2*l+1)/4/pi/l/(l+1));
                [BB, CC] = vsh_MS(l,ktheta);
                ms = reshape(-l:l,[],1);
                phase = exp(-1i.*ms.*reshape(kphi,1,[]));
                alm(l,1:size(BB,1)) = transpose(4*pi*cl*(-1).^ms.*(1i).^l    .* sum(phase .* conj(CC) .* reshape(pol_new_sph, 1, [], 3),2:3));
                blm(l,1:size(BB,1)) = transpose(4*pi*cl*(-1).^ms.*(1i).^(l-1).* sum(phase .* conj(BB) .* reshape(pol_new_sph, 1, [], 3),2:3));
            end
            % Normalize Coefficient
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
            if kx == 0 && ky == 0
                m_list = [-1 1];
            else
                m_list = -obj.lmax:obj.lmax;
            end
            for bulk = 1: obj.divide_section
                length_bulk = ceil(length(theta) / obj.divide_section);
                jj = (1+length_bulk * (bulk-1)) : min(length(theta), bulk*length_bulk);
                M_in = zeros(numel(jj),3);
                N_in = zeros(numel(jj),3);
                M_outh = zeros(numel(jj),3);
                N_outh = zeros(numel(jj),3);
                partial_theta = theta(jj);
                partial_rho_s = rho_s(jj);
                partial_rho = rho(jj);
                partial_out_flag = out_flag(jj);
                partial_in_flag = in_flag(jj);
                for l = 1:obj.lmax
                    [BB,CC,P] = vsh_MS(l,partial_theta);
                    BB = permute(BB,[2,3,1]);
                    CC = permute(CC,[2,3,1]);
                    P = permute(P,[2,3,1]);
                    cl=sqrt((2*l+1)/4/pi/l/(l+1));
                    % Spherical Bessel functions for internal field
                    j_s = sbesselj(l, partial_rho_s);j_s=j_s(:);j_s(isnan(j_s))=0;
                    dxi_s = ricbesjd(l, partial_rho_s);	dxi_s=dxi_s(:);dxi_s(isnan(dxi_s))=0;
                    j_rho_s = H_Rho(j_s,partial_rho_s,l);j_rho_s(isnan(j_rho_s))=0;
                    dxi_rho_s = Dxi_Rho(dxi_s,partial_rho_s,l);dxi_rho_s(isnan(dxi_rho_s))=0;
                    % Spherical Bessel functions for external field
                    h_m = sbesselh1(l, partial_rho);h_m=h_m(:);h_m(isnan(h_m))=0;
                    dxih_m = ricbesh1d(l, partial_rho);dxih_m=dxih_m(:);dxih_m(isnan(dxih_m))=0;
                    h_rho_m = H_Rho(h_m,partial_rho,l);h_rho_m(isnan(h_rho_m))=0;
                    dxih_rho_m = Dxi_Rho(dxih_m,partial_rho,l);dxih_rho_m(isnan(dxih_rho_m))=0;
                    for m = m_list
                        if abs(m) > l
                            continue
                        end
                        phase = exp(1i.*m.*phi(jj));
                        clc;
                        disp(['Volume section: ' num2str(bulk) ' / ' num2str(obj.divide_section)])
                        disp(['l: ' num2str(l) ' / ' num2str(obj.lmax)])
                        disp(['m: ' num2str(m)])
                        idx = m + l + 1;
                        M_in(:)   = j_s .* CC(:,:,idx);
                        N_in(:)   = l*(l+1).*j_rho_s.*P(:,:,idx) + dxi_rho_s .* BB(:,:,idx);

                        M_outh(:) = h_m .* CC(:,:,idx);
                        N_outh(:) = l*(l+1).*h_rho_m.*P(:,:,idx) + dxih_rho_m .* BB(:,:,idx);

                        E_scat_T(jj,:) = E_scat_T(jj,:) + (partial_out_flag .* (full(p_out(l*(l+1)+m)) .* M_outh + full(q_out(l*(l+1)+m)) .* N_outh) ... 
                                                        + partial_in_flag .* (full(p_in(l*(l+1)+m)) .* M_in + full(q_in(l*(l+1)+m)) .* N_in)) .* ((-1)^m * cl *phase);
                    end
                end
            end
            out_flag = reshape(out_flag,size(xf0));
            E_scat_T = sum(reshape(E_scat_T,[],1,3) .* reshape(Rx,[],3,3),3);
            E_scat_T = reshape(E_scat_T,[size(xf0), 3]);
            E_field_ref = current_source.generate_Efield(zeros(2,3));
            Efield = E_field_ref.*out_flag + E_field_ref(floor((size(E_field_ref,1)+1)/2),floor((size(E_field_ref,2)+1)/2),floor((size(E_field_ref,3)+1)/2),:).*E_scat_T;
            Efield = gather(Efield);
        end
    end
end


