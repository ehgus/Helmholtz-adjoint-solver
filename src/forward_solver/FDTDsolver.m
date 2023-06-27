classdef FDTDsolver < ForwardSolver
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
        % interative GUI
        hide_GUI = true;
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
        function obj=FDTDsolver(params)
            obj@ForwardSolver(params);
            % check boundary thickness
            if length(obj.boundary_thickness) == 1
                obj.boundary_thickness = zeros(1,3);
                obj.boundary_thickness(:) = obj.boundary_thickness;
            end
            assert(length(obj.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            obj.boundary_thickness_pixel = round(obj.boundary_thickness*obj.wavelength/obj.RI_bg./(obj.resolution.*2));
            obj.boundary_thickness_pixel(3) = max(1, obj.boundary_thickness_pixel(3)); % make place for source
            % find lumerical solver
            obj.lumerical_exe = FDTDsolver.find_lumerical_exe();
        end
        function set_RI(obj,RI)
            obj.RI=single(RI);%single computation are faster
            obj.initial_ZP_3=size(obj.RI,3);%size before adding boundary
            obj.condition_RI();%modify the RI (add padding and boundary)
            obj.init();%init the parameter for the forward model
        end
        function condition_RI(obj)
            %add boundary to the RI
            obj.create_boundary_RI();
        end
        function create_boundary_RI(obj)
            %% SHOULD BE REMOVED: boundary_thickness_pixel is the half from the previous definition
            old_RI_size=size(obj.RI);
            pott=RI2potential(obj.RI,obj.wavelength,obj.RI_bg);
            pott=padarray(pott,obj.boundary_thickness_pixel,'replicate');
            obj.RI=potential2RI(pott,obj.wavelength,obj.RI_bg);
            
            obj.ROI = [...
                obj.boundary_thickness_pixel(1)+1 obj.boundary_thickness_pixel(1)+old_RI_size(1)...
                obj.boundary_thickness_pixel(2)+1 obj.boundary_thickness_pixel(2)+old_RI_size(2)...
                obj.boundary_thickness_pixel(3)+1 obj.boundary_thickness_pixel(3)+old_RI_size(3)];
            obj.RI = potential2RI(pott,obj.wavelength,obj.RI_bg);
        end
        function init(obj)
            grid_size = size(obj.RI);
            warning('off','all');
            obj.utility=derive_utility(obj, grid_size); % the utility for the space with border
            warning('on','all');
        end
        function [Efield, Hfield]=solve(obj,input_field)
            input_field=single(input_field);
            assert(isfolder(obj.fdtd_temp_dir), 'FDTD temp folder is not valid')
            assert(size(input_field,3) == 2, 'The 3rd dimension of input_field should indicate polarization')
            if obj.verbose && size(input_field,3)==1
                warning('Input is scalar but scalar equation is less precise');
            end
            if size(input_field,3)>2
                error('Input field must be either a scalar or a 2D vector');
            end
            source_H = reshape([-1 1],1,1,2).*flip(input_field,3);
            
            
            %2D to 3D field
            [input_field] = obj.transform_field_3D(input_field);
            source_H = obj.transform_field_3D(source_H);
            %refocusing
            input_field=input_field.*exp(obj.utility.refocusing_kernel.*obj.resolution(3).*(-floor(obj.initial_ZP_3/2)-1-obj.padding_source));
            source_H=source_H.*exp(obj.utility.refocusing_kernel.*obj.resolution(3).*(-floor(obj.initial_ZP_3/2)-1-obj.padding_source));

            %normalisation needed here only for source terms (term from the helmotz equation in plane source because of double derivative)
            k_space_mask = sqrt(max(0,1-obj.utility.fourier_space.coor{1}.^2/obj.utility.k0_nm^2-obj.utility.fourier_space.coor{2}.^2/obj.utility.k0_nm^2));
            input_field=input_field.*k_space_mask;
            source_H=source_H.*k_space_mask;

            source=fftshift(ifft2(ifftshift(input_field)));
            source_H=fftshift(ifft2(ifftshift(source_H)));
            assert(isequal(size(source,1:2),size(obj.RI,1:2)),'Field and RI sizes are not consistent')
            
            %find the main component of the field
            phi=0;
            theta=0;
            para_pol=1;
            ortho_pol=0;
            [RI, roi] = lumerical_pad_RI(obj.RI, obj.ROI);
            resolution = obj.resolution .* [1e-6 1e-6 1e-6]; % um to meter (SI unit)
            roi = reshape(roi,2,3) .* reshape(resolution,1,3);
            lumerical_save_field(source,source_H,obj.resolution, fullfile(obj.fdtd_temp_dir, 'field.mat'));

            base_index = obj.RI_bg;
            lambda = obj.wavelength;
            is_plane_wave=double(obj.is_plane_wave);
            phi=double(phi);
            theta=double(theta);
            para_pol=double(para_pol);
            ortho_pol=double(ortho_pol);
            shutoff_min = double(obj.xtol);
            dt_stability_factor = double(obj.dt_stability_factor);
            pml_x = double(obj.PML_boundary(1));
            pml_y = double(obj.PML_boundary(2));
            pml_z = double(obj.PML_boundary(3));
            GUI_option = "";
            if obj.hide_GUI
                GUI_option = "-nw";
            end
            save(fullfile(obj.fdtd_temp_dir, 'optical.mat'),'lambda','base_index', 'RI', 'resolution', ...
                'roi','is_plane_wave','phi','theta','para_pol','ortho_pol','shutoff_min','dt_stability_factor','pml_x','pml_y','pml_z');
            assert(isfile(fullfile(obj.fdtd_temp_dir, "lumerical_fdtd_script.lsf")), sprintf("The script you tried to open does not exist on %s", obj.fdtd_temp_dir))
            command = sprintf('cd %s && "%s" -exit -run "lumerical_fdtd_script.lsf" %s', obj.fdtd_temp_dir, obj.lumerical_exe, GUI_option);
            system(command);

            % check the result data is still being written
            fid = fopen(fullfile(obj.fdtd_temp_dir, 'result.mat'),'r');
            while fid == -1
                pause(1);
                fid = fopen(fullfile(obj.fdtd_temp_dir, 'result.mat'),'r');
            end
            fclose(fid);
            
            %% End of FDTD
            load(fullfile(obj.fdtd_temp_dir, 'result.mat'),'res_vol','res_vol_H');

            Efield=reshape(res_vol.E,length(res_vol.x),length(res_vol.y),length(res_vol.z),3);
            Hfield=reshape(res_vol_H.H,length(res_vol_H.x),length(res_vol_H.y),length(res_vol_H.z),3);
            if obj.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Efield(:,floor(size(Efield,2)/2)+1,:))')));axis image; colorbar; axis off;drawnow
                colormap hot
            end
        end
    end
end

function [RI_pad, ROI_pad] = lumerical_pad_RI(RI, ROI)
    RI_pad=double(RI);
    RI_pad=cat(1,RI_pad,RI_pad,RI_pad);
    RI_pad=cat(2,RI_pad,RI_pad,RI_pad);
    ROI_pad = ROI;
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
