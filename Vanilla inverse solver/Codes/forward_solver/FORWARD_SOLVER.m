classdef (Abstract) FORWARD_SOLVER < handle
    properties
        % field information
        wavelength {mustBePositive} =1;     %wavelength [um]
        NA {mustBePositive} = 1;            %Numerical aperture of input/output waves
        vector_simulation logical = true;   %True/false: dyadic/scalar Green's function
        use_abbe_sine logical = true;       %Abbe sine condition according to demagnification condition
        utility
        % scattering object information: set_RI define these properties
        RI;                                 %Refractive index map
        RI_bg;                              %The representative refractive index
        resolution(1,3) = [1 1 1];           %3D Voxel size [um]
        % acceleration
        use_GPU logical = true;             %GPU acceleration
        % configuration
        verbose = false;                    %verbose option for narrative report
    end
    methods(Abstract)
        [E, transmitted_E, reflected_E] = solve(obj, input_field)
        set_RI(obj, RI) % determine RI and RI_bg
    end
    methods
        function obj=FORWARD_SOLVER(options)
            obj = obj.update_parameters(options);
        end
        function obj=update_parameters(obj, options)
            % Update parameters from default settings
            % It ignores unsupported fields
            instance_properties = intersect(properties(obj),fieldnames(options));
            for i = 1:length(instance_properties)
                property = instance_properties{i};
                obj.(property) = options.(property);
            end
        end
        function fft_Field_3pol=transform_field_3D(obj,fft_Field_2pol)
            Nsize = size(fft_Field_2pol);
            h.utility = derive_utility(obj, Nsize);
            fft_Field_2pol = fft_Field_2pol.*h.utility.NA_circle;
            
            if obj.use_abbe_sine
                %abbe sine condition is due to the magnification
                filter=single(h.utility.NA_circle);
                filter(h.utility.NA_circle)=filter(h.utility.NA_circle)./sqrt(h.utility.cos_theta(h.utility.NA_circle));
                fft_Field_2pol=fft_Field_2pol.*filter;
            end
            if Nsize(3)==2
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(Nsize, obj.RI_bg, obj.wavelength, obj.resolution);
                
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
        function fft_Field_2pol=transform_field_2D(obj,fft_Field_3pol) 
            Nsize=size(fft_Field_3pol);
            h.utility = derive_utility(obj, Nsize);
            fft_Field_3pol=fft_Field_3pol.*h.utility.NA_circle;
            
            if obj.use_abbe_sine
                %abbe sine condition is due to the magnification
                fft_Field_3pol=fft_Field_3pol.*sqrt(h.utility.cos_theta).*h.utility.NA_circle;
            end
            if size(fft_Field_3pol,3)>1
                if Nsize(3)~=3
                    error('Near field has three polarisation');
                end
                
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(Nsize, obj.RI_bg, obj.wavelength, obj.resolution);
                
                fft_Field_3pol=fft_Field_3pol.*K_mask;
                
                Field_new_basis=zeros(Nsize(1),Nsize(2),2,size(fft_Field_3pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_3pol         .*ewald_TanVec,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_3pol(:,:,1:2,:).*Perp_2D,3);
                
                fft_Field_2pol=zeros(Nsize(1),Nsize(2),2,size(fft_Field_3pol,4),'single');%the field in the 2D
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,1,:).*Radial_2D;
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,2,:).*Perp_2D;
            else
                fft_Field_2pol=fft_Field_3pol;
            end
        end
        function fft_Field_2pol=transform_field_2D_reflection(obj,fft_Field_3pol)
            Nsize=size(fft_Field_3pol);
            h.utility = derive_utility(obj, Nsize);
            fft_Field_3pol=fft_Field_3pol.*h.utility.NA_circle;
            
            if obj.use_abbe_sine
                %abbe sine condition is due to the magnification
                fft_Field_3pol=fft_Field_3pol.*sqrt(h.utility.cos_theta).*h.utility.NA_circle;
            end
            if size(fft_Field_3pol,3)>1
                assert(Nsize(3)==3, 'Near field has three polarisation')
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(Nsize, obj.RI_bg, obj.wavelength,obj.resolution);
                
                ewald_TanVec(:,:,3)=-ewald_TanVec(:,:,3);%because reflection invers k3
                
                fft_Field_3pol=fft_Field_3pol.*K_mask;
                
                Field_new_basis=zeros(Nsize(1),Nsize(2),2,size(fft_Field_3pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_3pol         .*ewald_TanVec,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_3pol(:,:,1:2,:).*Perp_2D,3);
                
                fft_Field_2pol=zeros(Nsize(1),Nsize(2),2,size(fft_Field_3pol,4),'single');%the field in the 2D
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,1,:).*Radial_2D;
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,2,:).*Perp_2D;
            else
                fft_Field_2pol=fft_Field_3pol;
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
            utility = DERIVE_OPTICAL_TOOL(params, obj.use_GPU);
        end
    end
end

function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(Nsize,n_m,lambda,dx)
    k0_nm =n_m/lambda; % [um-1, spatial wavenumber @ medium ]
    kres=1./dx(1:2)./Nsize(1:2);
    K_1=single(kres(1)/k0_nm*(-floor(Nsize(1)/2):Nsize(1)-floor(Nsize(1)/2)-1));%normalised to diffraction limit k1
    K_2=single(kres(2)/k0_nm*(-floor(Nsize(2)/2):Nsize(1)-floor(Nsize(2)/2)-1));%normalised to diffraction limit k2
    K_1=reshape(K_1,[],1);
    K_2=reshape(K_2,1,[]);
    K_3=sqrt(max(0,1-(K_1.^2+K_2.^2)));
    K_mask = K_3~=0;

    Radial_2D=zeros(Nsize(1),Nsize(2),2,'single');
    Perp_2D=zeros(Nsize(1),Nsize(2),2,'single');
    norm_rad=sqrt(K_1.^2+K_2.^2);

    temp1=K_1./norm_rad;
    temp2=K_2./norm_rad;
    temp1(norm_rad==0)=1;
    temp2(norm_rad==0)=0;

    Radial_2D(:,:,1)=temp1;
    Radial_2D(:,:,2)=temp2;
    clear temp1;
    clear temp2;

    Perp_2D(:,:,1)=Radial_2D(:,:,2);
    Perp_2D(:,:,2)=-Radial_2D(:,:,1);

    Radial_3D=zeros(Nsize(1),Nsize(2),3,'single');
    norm_sph=sqrt(K_1.^2+K_2.^2+K_3.^2);
    Radial_3D(:,:,1)=K_1./norm_sph;
    Radial_3D(:,:,2)=K_2./norm_sph;
    Radial_3D(:,:,3)=K_3./norm_sph;

    ewald_TanProj=sum(Radial_3D(:,:,1:2).*Radial_2D,3);
    ewald_TanVec=zeros(Nsize(1),Nsize(2),3,'single');

    ewald_TanVec(:,:,1:2)=Radial_2D(:,:,:);
    ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
    ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
    ewald_TanVec_norm(~K_mask)=1;
    ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;
end