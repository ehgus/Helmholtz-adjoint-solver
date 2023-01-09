classdef FORWARD_SOLVER_FDTD < FORWARD_SOLVER
    properties
        % scattering object w/ boundary
        boundary_thickness_pixel;
        padding_source = 1;
        boundary_thickness = 38;
        boundary_strength = 0.085;
        boundary_sharpness = 2;
        ROI;
        initial_ZP_3;
        % FDTD option
        xtol = 1e-5;
        dt_stability_factor = 0.99;
        is_plane_wave = false;
        PML_boundary = [false false true];
        % acceleration
        fdtd_temp_dir = fullfile('./FDTD_TEMP');
        lumerical_exe;
    end
    methods (Static)
        function lumerical_exe = find_lumerical_exe()
            assert(ispc, "The FDTD solver only supports Windows platform")
            solver_list = dir("C:/Program Files/Lumerical/v*/bin/fdtd-solutions.exe");
            assert(~isempty(solver_list), "The Lumerical solver is not found")
            
            solver_dir = sort({solver_list.folder});
            modern_solver = solver_dir{end};
            lumerical_exe = fullfile(modern_solver, 'fdtd-solutions.exe');
        end
    end

    methods
        function h=FORWARD_SOLVER_FDTD(params)
            h@FORWARD_SOLVER(params);
            % check boundary thickness
            if length(h.boundary_thickness) == 1
                h.boundary_thickness = zeros(1,3);
                h.boundary_thickness(:) = h.boundary_thickness;
            end
            assert(length(h.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            h.boundary_thickness_pixel = round(h.boundary_thickness*h.wavelength/h.RI_bg./(h.resolution.*2));
            h.boundary_thickness_pixel(3) = max(1, h.boundary_thickness_pixel(3)); % make place for source
            % find lumerical solver
            h.lumerical_exe = FORWARD_SOLVER_FDTD.find_lumerical_exe();
        end
        function set_RI(h,RI)
            h.RI=single(RI);%single computation are faster
            h.initial_ZP_3=size(h.RI,3);%size before adding boundary
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end
        function condition_RI(h)
            %add boundary to the RI
            h.create_boundary_RI();
        end
        function create_boundary_RI(h)
            %% SHOULD BE REMOVED: boundary_thickness_pixel is the half from the previous definition
            old_RI_size=size(h.RI);
            pott=RI2potential(h.RI,h.wavelength,h.RI_bg);
            pott=padarray(pott,h.boundary_thickness_pixel,'replicate');
            h.RI=potential2RI(pott,h.wavelength,h.RI_bg);
            
            h.ROI = [...
                h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+old_RI_size(1)...
                h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+old_RI_size(2)...
                h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+old_RI_size(3)];
            h.RI = potential2RI(pott,h.wavelength,h.RI_bg);
        end
        function init(h)
            Nsize = size(h.RI);
            warning('off','all');
            h.utility=derive_utility(h, Nsize); % the utility for the space with border
            warning('on','all');
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            input_field=single(input_field);
            
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
            source_H = reshape([-1 1],1,1,2).*flip(input_field,3);
            
            
            %2D to 3D field
            [input_field] = h.transform_field_3D(input_field);
            source_H = h.transform_field_3D(source_H);
            %refocusing
            input_field=input_field.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(-floor(h.initial_ZP_3/2)-1-h.padding_source));
            source_H=source_H.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(-floor(h.initial_ZP_3/2)-1-h.padding_source));
            source_0_3D=input_field;%save to remove the reflection

            %normalisation needed here only for source terms (term from the helmotz equation in plane source because of double derivative)
            k_space_mask = sqrt(max(0,1-h.utility.fourier_space.coor{1}.^2/h.utility.k0_nm^2-h.utility.fourier_space.coor{2}.^2/h.utility.k0_nm^2));
            input_field=input_field.*k_space_mask;
            source_H=source_H.*k_space_mask;

            input_field=fftshift(ifft2(ifftshift(input_field)));
            source_H=fftshift(ifft2(ifftshift(source_H)));
            %compute
            out_pol=1;
            if h.vector_simulation
                out_pol=2;
            end
            fields_trans=[];
            if h.return_transmission
                fields_trans=ones(size(h.RI,1),size(h.RI,2),out_pol,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.return_reflection
                fields_ref=ones(size(h.RI,1),size(h.RI,2),out_pol,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.return_3D
                fields_3D=ones(size(h.RI,1),size(h.RI,2),h.initial_ZP_3,size(input_field,3),size(input_field,4),'single');
            end
            
            for field_num=1:size(input_field,4)
                Field=h.solve_forward(input_field(:,:,:,field_num), source_H(:,:,:,field_num));
                %crop and remove near field (3D to 2D field)
                
                if h.return_3D
                    fields_3D(:,:,:,:,field_num)=Field;
                end
                if h.return_transmission
                    field_trans= squeeze(Field(:,:,end,:));
                    field_trans=fftshift(fft2(ifftshift(field_trans)));
                    [field_trans] = h.transform_field_2D(field_trans);
                    field_trans=field_trans.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(floor(h.initial_ZP_3/2)+1+h.padding_source-(h.initial_ZP_3+1+h.padding_source)));
                    field_trans=field_trans.*h.utility.NA_circle;%crop to the objective NA
                    field_trans=fftshift(ifft2(ifftshift(field_trans)));
                    fields_trans(:,:,:,field_num)=squeeze(field_trans);
                end
                if h.return_reflection
                    field_ref= squeeze(Field(:,:,1,:));
                    field_ref=fftshift(fft2(ifftshift(field_ref)));
                    field_ref=field_ref-source_0_3D(:,:,:,field_num).*exp(h.utility.refocusing_kernel.*h.resolution(3).*(+h.padding_source));
                    [field_ref] = h.transform_field_2D_reflection(field_ref);
                    field_ref=field_ref.*exp(h.utility.refocusing_kernel.*h.resolution(3).*(-floor(h.initial_ZP_3/2)-1));
                    field_ref=field_ref.*h.utility.NA_circle;%crop to the objective NA
                    field_ref=fftshift(ifft2(ifftshift(field_ref)));
                    fields_ref(:,:,:,field_num)=squeeze(field_ref);
                end
            end
            
        end
        function Field=solve_forward(h,source,source_H)
            assert(isequal(size(source,1:2),size(h.RI,1:2)),'Field and RI sizes are not consistent')
            assert(isfolder(h.fdtd_temp_dir), 'FDTD temp folder is not valid')
            %find the main component of the field
            Field_SPEC_ABS= sqrt(sum(abs(source).^2,3));
            [~,I]=max(Field_SPEC_ABS,[],'all','linear');
            phi=0;
            theta=0;
            para_pol=1;
            ortho_pol=0;
            [RI, roi] = lumerical_pad_RI(h.RI, h.ROI);
            resolution = h.resolution .* [1e-6 1e-6 1e-6]; % um to meter (SI unit)
            roi = reshape(roi,2,3) .* reshape(resolution,1,3);
            lumerical_save_field(source,source_H,h.resolution, fullfile(h.fdtd_temp_dir, 'field.mat'));

            return_trans=double(1);
            return_ref=double(1);
            return_vol=double(1);
            base_index = h.RI_bg;
            lambda = h.wavelength;
            is_plane_wave=double(h.is_plane_wave);
            phi=double(phi);
            theta=double(theta);
            para_pol=double(para_pol);
            ortho_pol=double(ortho_pol);
            shutoff_min = double(h.xtol);
            dt_stability_factor = double(h.dt_stability_factor);
            pml_x = double(h.PML_boundary(1));
            pml_y = double(h.PML_boundary(2));
            pml_z = double(h.PML_boundary(3));

            save(fullfile(h.fdtd_temp_dir, 'optical.mat'),'lambda','base_index', 'RI', 'resolution', ...
                'roi', 'return_trans','return_ref','return_vol',...
                'is_plane_wave','phi','theta','para_pol','ortho_pol','shutoff_min','dt_stability_factor','pml_x','pml_y','pml_z');
            assert(isfile(fullfile(h.fdtd_temp_dir, "lumerical_fdtd_script.lsf")), sprintf("The script you tried to open does not exist on %s", h.fdtd_temp_dir))
            command = sprintf('cd %s && "%s" -exit -run "lumerical_fdtd_script.lsf"', h.fdtd_temp_dir, h.lumerical_exe);
            system(command);

            % check the result data is still being written
            fid = fopen(fullfile(h.fdtd_temp_dir, 'result.mat'),'r');
            while fid == -1
                pause(1);
                fid = fopen(fullfile(h.fdtd_temp_dir, 'result.mat'),'r');
            end
            fclose(fid);
            
            %% End of FDTD
            load(fullfile(h.fdtd_temp_dir, 'result.mat'),'res_vol');

            Field=reshape(res_vol.E,length(res_vol.x),length(res_vol.y),length(res_vol.z),3);
            if h.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))')));axis image; title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]); colorbar; axis off;drawnow
                colormap hot
            end
        end
    end
end

function [RI_pad, ROI_pad] = lumerical_pad_RI(RI, ROI)
    RI_pad = padarray(double(RI), [10 10 0], 'circular'); % mitigate the boundary effect
    RI_pad = padarray(RI_pad, [0 0 1],'replicate');     % prepare a region to place a illumination source
    ROI_pad = ROI;
    ROI_pad(1:4) = ROI_pad(1:4) + 10;
    ROI_pad(5:6)  = ROI_pad(5:6) + 1;
end

function lumerical_save_RI_text(RI,dx,file_name)
%% Lumerical function: lumerical_save_RI_text

RI=single(RI);
reality=isreal(RI);
RI=cat(1,RI,RI,RI);
RI=cat(2,RI,RI,RI);

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
    fprintf(fid, '%.10g %.10g\n', [real(RI(:))'; imag(RI(:)')] );
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
