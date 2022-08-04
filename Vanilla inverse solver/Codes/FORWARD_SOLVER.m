classdef FORWARD_SOLVER < handle
    properties (SetAccess = protected, Hidden = true)
        parameters;
        RI;
        
        utility;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            %SIMULATION PARAMETERS
            params.return_transmission=true;%return transmission field
            params.return_reflection=false;%return reflection field
            params.return_3D=false;%return 3D field
            params.use_GPU=true;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER(params)
            h.parameters=params;
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters,h.parameters.use_GPU);%reset it to have the gpu used
            warning('on','all');
        end
        function set_RI(h,RI)
            h.RI=RI;
        end
        function [field_trans,field_ref,field_3D]=solve(h,input_field)
            error("You need to specify the forward solver to solve");
            field_trans=[];
            field_ref=[];
            field_3D=[];
        end
        function fft_Field_3pol=transform_field_3D(h,fft_Field_2pol)
            ZP=size(fft_Field_2pol,1:4);
            fft_Field_2pol=fft_Field_2pol.*h.utility.NA_circle;
            
            if h.parameters.use_abbe_sine
                %abbe sine condition is due to the magnification
                
                filter=single(h.utility.NA_circle);
                filter(h.utility.NA_circle)=filter(h.utility.NA_circle)./sqrt(h.utility.cos_theta(h.utility.NA_circle));
                fft_Field_2pol=fft_Field_2pol.*filter;
            end
            if ZP(3)==2
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,h.parameters.RI_bg,h.parameters.wavelength,h.parameters.resolution);
                
                fft_Field_2pol=fft_Field_2pol.*K_mask;
                
                Field_new_basis=zeros(ZP(1),ZP(2),2,ZP(4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_2pol.*Radial_2D,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_2pol.*Perp_2D,3);
                fft_Field_3pol=zeros(ZP(1),ZP(2),3,ZP(4),'single');%the field in the 3D
                fft_Field_3pol         =fft_Field_3pol          + Field_new_basis(:,:,1,:).*ewald_TanVec;
                fft_Field_3pol(:,:,1:2,:)=fft_Field_3pol(:,:,1:2,:) + Field_new_basis(:,:,2,:).*Perp_2D;
            elseif ZP(3)==1
                fft_Field_3pol=fft_Field_2pol;
            else
                error('Far field has two polarisation');
            end
        end

        function fft_Field_2pol=transform_field_2D(h,fft_Field_3pol,reflection_mode)
            ZP=size(fft_Field_3pol,1:4);
            fft_Field_3pol=fft_Field_3pol.*h.utility.NA_circle;
            
            if h.parameters.use_abbe_sine
                %abbe sine condition is due to the magnification
                
                fft_Field_3pol=fft_Field_3pol.*sqrt(h.utility.cos_theta).*h.utility.NA_circle;
            end
            if ZP(3)==3
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,h.parameters.RI_bg,h.parameters.wavelength,h.parameters.resolution);
                if reflection_mode
                    ewald_TanVec(:,:,3)=-ewald_TanVec(:,:,3);%because reflection invers k3
                end
                
                fft_Field_3pol=fft_Field_3pol.*K_mask;
                Field_new_basis=zeros(ZP(1),ZP(2),2,ZP(4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_3pol         .*ewald_TanVec,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_3pol(:,:,1:2,:).*Perp_2D,3);
                
                fft_Field_2pol=zeros(ZP(1),ZP(2),2,ZP(4),'single');%the field in the 2D
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,1,:).*Radial_2D;
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,2,:).*Perp_2D;
            elseif ZP(3)==1
                fft_Field_2pol=fft_Field_3pol;
            else
                error('Near field has three polarisation or scalar');
            end
        end
    end
end
function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,n_m,lambda,dx)
size_1 = ZP(1);
size_2 = ZP(2);

k =2*pi*n_m/lambda; % [um-1, spatial wavenumber @ medium ]

kres=1./dx(1:2)./ZP(1:2);

K_1=single(2*pi*kres(1)/k*cat(2,0:floor((size_1-1)/2),floor(size_1/2):-1:1));%normalised to diffraction limit k1
K_2=single(2*pi*kres(2)/k*cat(2,0:floor((size_2-1)/2),floor(size_2/2):-1:1));%normalised to diffraction limit k2

K_1=reshape(K_1,[],1);
K_2=reshape(K_2,1,[]);

K_3=real(sqrt(1-(K_1.^2+K_2.^2)));
K_mask = ~(K_3==0);

Radial_3D=zeros(size_1,size_2,3,'single');
Radial_2D=zeros(size_1,size_2,2,'single');
Perp_2D=zeros(size_1,size_2,2,'single');
norm_sph=sqrt(K_1.^2+K_2.^2+K_3.^2);
norm_rad=sqrt(K_1.^2+K_2.^2);

Radial_3D(:,:,1)=K_1./norm_sph;
Radial_3D(:,:,2)=K_2./norm_sph;
Radial_3D(:,:,3)=K_3./norm_sph;
Radial_3D(1,1,:)=1/sqrt(3);

Radial_2D(:,:,1)=K_1./norm_rad;
Radial_2D(:,:,2)=K_2./norm_rad;
Radial_2D(1,1,:)=1/sqrt(2);

Perp_2D(:,:,1)=Radial_2D(:,:,2);
Perp_2D(:,:,2)=-Radial_2D(:,:,1);

ewald_TanVec=zeros(size_1,size_2,3,'single');
ewald_TanProj=sum(Radial_3D(:,:,[1,2]).*Radial_2D,3);
ewald_TanVec(:,:,[1,2])=Radial_2D;
ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
ewald_TanVec_norm(~K_mask)=1;
ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;

end