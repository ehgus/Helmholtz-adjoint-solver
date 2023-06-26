classdef (Abstract) ForwardSolver < OpticalSimulation
    properties
        % Additional field information
        NA {mustBePositive} = 1;            %Numerical aperture of input/output waves
        use_abbe_sine logical = true;       %Abbe sine condition according to demagnification condition
        utility
        % Addtional scattering object information: set_RI define these properties
        RI_bg;                              %The representative refractive index
        % acceleration
        use_GPU logical = true;             %GPU acceleration
        % return values
        return_3D = true;
        return_transmission = true;
        return_reflection = true;
    end
    methods(Abstract)
        [E, transmitted_E, reflected_E] = solve(obj, input_field)
        set_RI(obj, RI) % determine RI and RI_bg
    end
    methods
        function obj=ForwardSolver(options)
            obj@OpticalSimulation(options);
        end
        function fft_Field_3pol=transform_field_3D(obj,fft_Field_2pol)
            Nsize = size(fft_Field_2pol);
            utils = derive_utility(obj, Nsize);
            fft_Field_2pol = fft_Field_2pol.*utils.NA_circle;
            
            if obj.use_abbe_sine
                %abbe sine condition is due to the magnification
                filter=single(utils.NA_circle);
                filter(utils.NA_circle)=filter(utils.NA_circle)./sqrt(utils.cos_theta(utils.NA_circle));
                fft_Field_2pol=fft_Field_2pol.*filter;
            end
            if Nsize(3)==2
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(utils);
                
                fft_Field_2pol=fft_Field_2pol.*K_mask;
                
                Field_new_basis=zeros(Nsize(1),Nsize(2),2,size(fft_Field_2pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_2pol.*Radial_2D,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_2pol.*Perp_2D,3);
                
                fft_Field_3pol=zeros(Nsize(1),Nsize(2),3,size(fft_Field_2pol,4),'single');%the field in the 3D
                fft_Field_3pol         =fft_Field_3pol          + Field_new_basis(:,:,1,:).*ewald_TanVec;
                fft_Field_3pol(:,:,1:2,:)=fft_Field_3pol(:,:,1:2,:) + Field_new_basis(:,:,2,:).*Perp_2D;
            elseif Nsize(3)==1
                fft_Field_3pol=fft_Field_2pol;
            else
                error('Far field has two polarisation');
            end
        end
        function utility = derive_utility(obj, Nsize)
            params = struct( ...
                'resolution', obj.resolution ,...
                'size', Nsize, ...
                'wavelength', obj.wavelength, ...
                'RI_bg', obj.RI_bg, ...
                'NA', obj.NA ...
            );
            utility = derive_optical_tool(params, obj.use_GPU);
        end
    end
end

function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(utility)
    % utility to convert (r, theta) -> (x,y,z)
    K_1=utility.fourier_space.coor{1};
    K_2=utility.fourier_space.coor{2};
    K_3=utility.k3;
    K_mask = K_3 > 0;
    Nsize = utility.size;
    % need to consider k0_nm coefficient 
    Radial_2D =    zeros(Nsize(1),Nsize(2),2,'single');
    Perp_2D =      zeros(Nsize(1),Nsize(2),2,'single');
    Radial_3D =    zeros(Nsize(1),Nsize(2),3,'single');
    ewald_TanVec = zeros(Nsize(1),Nsize(2),3,'single');
    norm_rad=utility.fourier_space.coorxy;

    Radial_2D(:,:,1)=K_1./norm_rad;
    Radial_2D(:,:,2)=K_2./norm_rad;
    Radial_2D(K_1 == 0,K_2 == 0,:) = [1 0]; % define the center of radial

    Perp_2D(:,:,1)=Radial_2D(:,:,2);
    Perp_2D(:,:,2)=-Radial_2D(:,:,1);

    Radial_3D(:,:,1)=K_1/utility.k0_nm.*K_mask;
    Radial_3D(:,:,2)=K_2/utility.k0_nm.*K_mask;
    Radial_3D(:,:,3)=K_3/utility.k0_nm.*K_mask;

    ewald_TanProj=sum(Radial_3D(:,:,1:2).*Radial_2D,3);
    ewald_TanVec(:,:,1:2)=Radial_2D(:,:,:);
    ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
    ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
    ewald_TanVec_norm(~K_mask)=1;
    ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;
end