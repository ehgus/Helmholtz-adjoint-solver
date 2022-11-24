classdef FORWARD_SOLVER_FDTD < FORWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        Bornmax;
        boundary_thickness_pixel;
        ROI;
        
        Greenp;
        rads;
        psi;
        PSI;
        F;
        eta;
        eye_3;
        
        kern1;
        kern2;
        
        V;
        initial_ZP_3;
        pole_num;
        green_absorbtion_correction;
        eps_imag;

        fdtd_temp_dir;
    end
    methods
        function get_default_parameters(h)
            get_default_parameters@FORWARD_SOLVER(h);
            %specific parameters
            h.parameters.iterations_number=-1;
            h.parameters.use_GPU=true;
            h.parameters.use_cuda=false;
            h.parameters.padding_source=1;
            h.parameters.boundary_strength =0.085;
            h.parameters.boundary_thickness = 38;
            h.parameters.boundary_sharpness = 2;
            h.parameters.verbose = false;
            h.parameters.non_cyclic_conv=true;
            h.parameters.xtol = 1e-5;
            h.parameters.dt_stability_factor = 0.99;
            h.fdtd_temp_dir = './FDTD_TEMP';
        end

        function h=FORWARD_SOLVER_FDTD(params)
            h@FORWARD_SOLVER(params);
        end
        function set_RI(h,RI)
            RI=single(RI);%single computation are faster
            set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            
            h.initial_ZP_3=size(h.RI,3);%size before adding boundary
            
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end
        function condition_RI(h)
            %add boundary to the RI
            h.RI = cat(3,h.parameters.RI_bg * ones(size(h.RI,1),size(h.RI,2),h.parameters.padding_source+1,size(h.RI,4),size(h.RI,5),'single'),h.RI);%add one slice to put the source and one to get the reflection
            h.RI = cat(3,h.RI,h.parameters.RI_bg * ones(size(h.RI,1),size(h.RI,2),1,size(h.RI,4),size(h.RI,5),'single'));%add one slice to retrive the RI
            h.create_boundary_RI();
            %update the size in the parameters
            h.parameters.size=size(h.RI);
        end
        function create_boundary_RI(h)
            
            h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)/h.parameters.resolution(3));
            
            abs_scatt_pott = RI2potential((h.parameters.RI_bg+1i*h.parameters.boundary_strength),h.parameters.wavelength,h.parameters.RI_bg);
            
            if (h.parameters.use_GPU)
                h.RI = gpuArray(h.RI);
                h.RI = cat(3,h.RI,h.parameters.RI_bg.*ones(size(h.RI,1),size(h.RI,2),h.boundary_thickness_pixel,'single','gpuArray'));
            else
                h.RI = cat(3,h.RI,h.parameters.RI_bg.*ones(size(h.RI,1),size(h.RI,2),h.boundary_thickness_pixel,'single'));
            end
            V_temp = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
            
            x=(1:size(V_temp,3))-floor(size(V_temp,3)/2);x=circshift(x,-floor(size(V_temp,3)/2));
            x=x/(h.boundary_thickness_pixel/2);
            x=circshift(x,size(V_temp,3)-round(h.boundary_thickness_pixel/2));
            val=(exp(1./(abs(x).^(2*round(h.parameters.boundary_sharpness))-1)));val(abs(x)>=1)=0;
            
            abs_profile=val.*abs_scatt_pott;
            
            V_temp=V_temp+1i*imag(reshape((abs_profile),1,1,[]));
            
            h.RI = potential2RI(V_temp,h.parameters.wavelength,h.parameters.RI_bg);
            if (h.parameters.use_GPU)
                h.RI=gather(h.RI);
            end
        end
        function init(h)
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            
            
            if h.parameters.verbose && h.parameters.iterations_number>0
                warning('Best is to set iterations_number to -n for an automatic choice of this so that reflection to the ordern n-1 are taken in accound (transmission n=1, single reflection n=2, higher n=?)');
            end
            
            if ~h.parameters.use_GPU
                h.parameters.use_cuda=false;
            else
                h.RI=single(gpuArray(h.RI));
            end
            h.pole_num=1;
            if h.parameters.vector_simulation
                h.pole_num=3;
            end
            
            if ~h.parameters.use_cuda
                h.rads=...
                    (h.utility.fourier_space.coor{1}./h.utility.k0_nm).*reshape([1 0 0],1,1,1,[])+...
                    (h.utility.fourier_space.coor{2}./h.utility.k0_nm).*reshape([0 1 0],1,1,1,[])+...
                    (h.utility.fourier_space.coor{3}./h.utility.k0_nm).*reshape([0 0 1],1,1,1,[]);
            end
            
            h.V =real(h.RI);
            
            h.eps_imag = max(abs(h.V(:))).*1.01;
            
            h.green_absorbtion_correction=((2*pi*h.utility.k0_nm)^2)/((2*pi*h.utility.k0_nm)^2+1i.*h.eps_imag);
            
            step = abs(2*(2*pi*h.utility.k0_nm)/h.eps_imag);
            Bornmax_opt = ceil((h.initial_ZP_3 + h.parameters.padding_source )*h.parameters.resolution(3) / step + 1)*2;
            h.Bornmax = 0;
            
            if h.parameters.iterations_number==0
                error('set iterations_number to either a positive or negative value');
            elseif h.parameters.iterations_number<=0
                h.Bornmax=Bornmax_opt*abs(h.parameters.iterations_number);
            else
                h.Bornmax =h.parameters.iterations_number;
            end
            
            if h.parameters.verbose
                display(['number of step : ' num2str(h.Bornmax)])
            end
            
            h.psi = repmat(h.V,1,1,1,h.pole_num)*0; h.PSI = h.psi*0;
            
            
            if ~h.parameters.use_cuda
                h.eye_3=reshape(eye(3),1,1,1,3,3);
                if h.parameters.use_GPU
                    h.eye_3=gpuArray(h.eye_3);
                end
                
                h.Greenp = 1 ./ (4*pi^2.*abs(...
                    h.utility.fourier_space.coor{1}.^2 + ...
                    h.utility.fourier_space.coor{2}.^2 + ...
                    h.utility.fourier_space.coor{3}.^2 ...
                    )-(2*pi*h.utility.k0_nm)^2-1i*h.eps_imag);
                
                
                if h.parameters.use_GPU
                    h.rads = gpuArray(single(h.rads));
                    h.Greenp = gpuArray(single(h.Greenp));
                end
                
                h.Greenp=ifftshift(ifftshift(ifftshift(h.Greenp,1),2),3);
                h.rads=ifftshift(ifftshift(ifftshift(h.rads,1),2),3);
            end
            
            if h.parameters.verbose
                figure('units','normalized','outerposition',[0 0 1 1])
                colormap hot
            end
            
            h.RI=gather(h.RI);
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            if ~h.parameters.use_GPU
                h.parameters.use_cuda=false;
                input_field=single(input_field);
            else
                h.RI=single(gpuArray(h.RI));
                input_field=single(gpuArray(input_field));
            end
            if h.parameters.use_cuda
               if h.parameters.non_cyclic_conv 
                   error('not implemented');
               end
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
            source_H = reshape([-1 1],1,1,2).*flip(input_field,3);
            
            
            %2D to 3D field
            [input_field] = h.transform_field_3D(input_field);
            source_H = h.transform_field_3D(source_H);
            %refocusing
            input_field=input_field.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(-floor(h.initial_ZP_3/2)-1-h.parameters.padding_source));
            source_H=source_H.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(-floor(h.initial_ZP_3/2)-1-h.parameters.padding_source));
            source_0_3D=input_field;%save to remove the reflection
            k_space_mask = sqrt(max(0,1-(h.utility.fourier_space.coor{1}./h.utility.k0_nm).^2-(h.utility.fourier_space.coor{2}./h.utility.k0_nm).^2));
            input_field=input_field.*k_space_mask;%normalisation needed here only for source terms (term from the helmotz equation in plane source because of double derivative)
            source_H=source_H.*k_space_mask;%normalisation needed here only for source terms (term from the helmotz equation in plane source because of double derivative)

            input_field=fftshift(ifft2(ifftshift(input_field)));
            source_H=fftshift(ifft2(ifftshift(source_H)));
            %compute
            out_pol=1;
            if h.pole_num==3
                out_pol=2;
            end
            fields_trans=[];
            if h.parameters.return_transmission
                fields_trans=ones(size(h.V,1),size(h.V,2),out_pol,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.parameters.return_reflection
                fields_ref=ones(size(h.V,1),size(h.V,2),out_pol,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.parameters.return_3D
                fields_3D=ones(size(h.V,1),size(h.V,2),h.initial_ZP_3,size(input_field,3),size(input_field,4),'single');
            end
            
            for field_num=1:size(input_field,4)
                Field=h.solve_raw(input_field(:,:,:,field_num), source_H(:,:,:,field_num));
                %crop and remove near field (3D to 2D field)
                
                if h.parameters.return_3D
                    field_3D= Field(:,:,1:end-h.boundary_thickness_pixel,:);
                    fields_3D(:,:,:,:,field_num)=field_3D;
                end
                if h.parameters.return_transmission
                    field_trans= Field(:,:,end-h.boundary_thickness_pixel,:);
                    field_trans=squeeze(field_trans);
                    field_trans=fftshift(fft2(ifftshift(field_trans)));
                    [field_trans] = h.transform_field_2D(field_trans);
                    field_trans=field_trans.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(floor(h.initial_ZP_3/2)+1+h.parameters.padding_source-(h.initial_ZP_3+1+h.parameters.padding_source)));
                    field_trans=field_trans.*h.utility.NA_circle;%crop to the objective NA
                    field_trans=fftshift(ifft2(ifftshift(field_trans)));
                    fields_trans(:,:,:,field_num)=squeeze(field_trans);
                end
                if h.parameters.return_reflection
                    field_ref= Field(:,:,1+h.parameters.padding_source,:);
                    field_ref=squeeze(field_ref);
                    field_ref=fftshift(fft2(ifftshift(field_ref)));
                    field_ref=field_ref-source_0_3D(:,:,:,field_num).*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(+h.parameters.padding_source));
                    [field_ref] = h.transform_field_2D_reflection(field_ref);
                    field_ref=field_ref.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(-floor(h.initial_ZP_3/2)-1));
                    field_ref=field_ref.*h.utility.NA_circle;%crop to the objective NA
                    field_ref=fftshift(ifft2(ifftshift(field_ref)));
                    fields_ref(:,:,:,field_num)=squeeze(field_ref);
                end
            end
            
        end
        function Field=solve_raw(h,source,source_H)
            assert(isequal(size(source,1:2),size(h.RI,1:2)),'Field and RI sizes are not consistent')
            assert(isfolder(h.fdtd_temp_dir), 'FDTD temp folder is not valid')

            cd_lumerical = 'C:\Program Files\Lumerical\v212\bin\fdtd-solutions.exe';
            source_normalizer = -1i*4*pi*h.parameters.RI_bg/h.parameters.wavelength/h.parameters.resolution(3);
            source=source*source_normalizer;%normalise the source term
            source_H=source_H*source_normalizer;%normalise the source term

            %find the main component of the field
            Field_SPEC_ABS= sqrt(sum(abs(source).^2,3));
            [M,I]=max(Field_SPEC_ABS,[],'all','linear');
            Field_SPEC_ABS_OTHER=Field_SPEC_ABS;
            Field_SPEC_ABS_OTHER(I)=0;
            is_plane_wave = any(Field_SPEC_ABS_OTHER(:)<(M/100));
            phi=0;
            theta=0;
            para_pol=0;
            ortho_pol=0;

            is_plane_wave=false;

            if is_plane_wave
               disp('Simulating using plane wave illumination')
               [d1,d2]= ind2sub(size(Field_SPEC_ABS),I);
               d1=d1-(1+floor(size(Field_SPEC_ABS,1)/2));
               d2=d2-(1+floor(size(Field_SPEC_ABS,2)/2));

               kres2 = h.utility.fourier_space.coor{1};
               max_angle = h.parameters.RI_bg/h.parameters.wavelength/kres2;

               phi=angle(d1+1i*d2);
               theta=asin(sqrt(d1.^2+d2.^2)/max_angle);

               director=[cos(phi)*cos(theta) sin(phi)*cos(theta) sin(theta)];
               norm_director=[cos(phi+pi/2)*cos(theta) sin(phi+pi/2)*cos(theta) 0*sin(theta)];
               para_pol=sum(squeeze(source(1,1,:)).*squeeze(director'),'all');
               ortho_pol=sum(squeeze(source(1,1,:)).*squeeze(norm_director'),'all');

            end
            
            RImap=padarray(h.V,[1 1 0],'circular','post');
            source=padarray(source,[1 1 0],'circular','post');
            source_H=padarray(source_H,[1 1 0],'circular','post');
            
            size(RImap)


            lumerical_save_RI_text(RImap,h.parameters.resolution,fullfile(h.fdtd_temp_dir, 'index.txt'));
            lumerical_save_field(source,source_H,h.parameters.resolution, fullfile(h.fdtd_temp_dir, 'field.mat'));
            %bool is not supported by
            return_trans=double(1);
            return_ref=double(1);
            return_vol=double(1);
            base_index = h.parameters.RI_bg;
            lambda = h.parameters.wavelength;
            is_plane_wave=double(is_plane_wave);
            phi=double(phi);
            theta=double(theta);
            para_pol=double(para_pol);
            ortho_pol=double(ortho_pol);
            shutoff_min = double(h.parameters.xtol);
            dt_stability_factor = double(h.parameters.dt_stability_factor);

            save(fullfile(h.fdtd_temp_dir, 'optical.mat'),'lambda','base_index','return_trans','return_ref','return_vol',...
                'is_plane_wave','phi','theta','para_pol','ortho_pol','shutoff_min','dt_stability_factor');

            command=[' "' cd_lumerical '"      -run -exit  "' h.fdtd_temp_dir '\lumerical_fdtd_script.lsf" '];
            system(command);

            % check the result data is still being written
            fid = fopen(fullfile(h.fdtd_temp_dir, 'result.mat'),'r');
            while fid == -1
                pause(1);
                fid = fopen(fullfile(h.fdtd_temp_dir, 'result.mat'),'r');
            end
            fclose(fid);
            
            %% End of FDTD
            load(fullfile(h.fdtd_temp_dir, 'result.mat'));

            if true
                Field=reshape(res_vol.E,length(res_vol.x),length(res_vol.y),length(res_vol.z),3);
                Field=Field(1:end-1,1:end-1,:,:);
            end
            if h.parameters.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(Bornmax)]), colorbar, axis off,drawnow
                colormap hot
            end
        end
    end
end

function lumerical_save_RI_text(RI,dx,file_name)
%% Lumerical function: lumerical_save_RI_text

RI=single(RI);
reality=isreal(RI);

if length(dx)==1
    dx=dx*[1 1 1];
end

dx=dx*1e-6;%um to si units

fid=fopen(file_name,'W');
%set the parameters
for dim=1:3
    fprintf(fid, '%.10g %.10g %.10g \n', [single(size(RI,dim)) single(0)*dx(dim) single((size(RI,dim)-1)*dx(dim))]');
end
%set the data
if reality
    fprintf(fid, '%.10g\n', RI(:));
else
    fprintf(fid, '%.10g %.10g\n', real(RI(:)) , imag(RI(:)) );
end
fclose(fid);
end
function lumerical_save_field(FIELD,FIELD_H,dx,file_name)
%% Lumerical function: lumerical_save_field

FIELD=double(FIELD);
FIELD_H=double(FIELD_H);

if length(size(FIELD))~=3
   error('the field must be of size x by y by 3'); 
end
if size(FIELD,3)~=3
   error('the field must be of size x by y by 3'); 
end

if length(dx)==1
    dx=dx*[1 1 1];
end

dx=dx*1e-6;%um to si units

EM=struct;
EM.E=reshape(double(FIELD),[],3);

EM.Lumerical_dataset=struct;
EM.H=reshape(double(FIELD_H),[],3);
EM.Lumerical_dataset.attributes=[struct,struct]';
EM.Lumerical_dataset.attributes(1).variable='E';
EM.Lumerical_dataset.attributes(1).name='E';
EM.Lumerical_dataset.attributes(2).variable='H';
EM.Lumerical_dataset.attributes(2).name='H';

EM.Lumerical_dataset.geometry='rectilinear';
EM.x=reshape(double((0:(size(FIELD,1)-1))*dx(1)),[],1);
EM.y=reshape(double((0:(size(FIELD,2)-1))*dx(2)),[],1);
EM.z=0;

save(file_name,'EM')

end
