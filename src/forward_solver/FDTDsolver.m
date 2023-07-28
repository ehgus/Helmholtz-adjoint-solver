classdef FDTDsolver < ForwardSolver
    properties
        % scattering object w/ boundary
        boundary_thickness_pixel
        boundary_thickness
        ROI
        % FDTD option
        is_plane_wave = false
        PML_boundary = [false false true]
        % interative GUI
        hide_GUI = true
        % save directory
        save_directory = tempdir()
    end
    properties(Hidden = true)
        lumerical_api_dir
        lumerical_session
    end
    methods (Static)
        function lumerical_api_dir = find_lumerical_api()
            assert(ispc, "The FDTD solver only supports Windows platform")
            if ispc
                api_list = dir("C:/Program Files/Lumerical/v*/api/matlab");
                assert(~isempty(api_list), "The Lumerical solver is not found")
                api_dir = sort({api_list.folder});
                lumerical_api_dir = api_dir{end};
            else
                error("Current OS is not supported")
            end
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
            % find lumerical solverRI
            obj.lumerical_api_dir = FDTDsolver.find_lumerical_api();
            addpath(obj.lumerical_api_dir);
            if obj.hide_GUI
                obj.lumerical_session = appopen('fdtd', '-hide');
            else
                obj.lumerical_session = appopen('fdtd');
            end
        end
        function delete(obj)
            addpath(obj.lumerical_api_dir);
            appclose(obj.lumerical_session);
        end
        function set_RI(obj,RI)
            obj.RI= RI;
            obj.create_boundary_RI();%modify the RI (add padding and boundary)
        end
        function create_boundary_RI(obj)
            %% SHOULD BE REMOVED: boundary_thickness_pixel is the half from the previous definition
            old_RI_size=size(obj.RI);
            pott=RI2potential(obj.RI,obj.wavelength,obj.RI_bg);
            pott=padarray(pott,double(obj.boundary_thickness_pixel),'replicate');
            obj.RI=potential2RI(pott,obj.wavelength,obj.RI_bg);
            
            obj.ROI = [...
                obj.boundary_thickness_pixel(1)+1 obj.boundary_thickness_pixel(1)+old_RI_size(1)...
                obj.boundary_thickness_pixel(2)+1 obj.boundary_thickness_pixel(2)+old_RI_size(2)...
                obj.boundary_thickness_pixel(3)+1 obj.boundary_thickness_pixel(3)+old_RI_size(3)];
            obj.RI = potential2RI(pott,obj.wavelength,obj.RI_bg);
            % replicate along the periodic axis
            rep_size = ones(1,3);
            rep_size(~obj.PML_boundary) = 3;
            obj.RI = repmat(obj.RI, rep_size);
            obj.RI= double(obj.RI); % Lumerical only accept double-type variables
        end
        function [Efield, Hfield]=solve(obj,current_source)
            assert(isa(current_source, 'PlaneSource'), "PlaneSource is the only available source")
            addpath(obj.lumerical_api_dir);
            axis_name_list = {'x', 'y', 'z'};
            microns = 1e-6;
            resolution = obj.resolution .* microns;
            lumerical_roi = obj.ROI - 1;
            for axis = 1:3
                lumerical_roi([2*axis-1, 2*axis]) = lumerical_roi([2*axis-1, 2*axis]) * resolution(axis);
            end
            % initialize solver
            init_code = strcat('switchtolayout;', ...
                               'closeall;', ...
                               'deleteall;', ...
                               'clear;');
            appevalscript(obj.lumerical_session, init_code);
            for axis = 1:3
                appputvar(obj.lumerical_session, strcat('d', axis_name_list{axis}), resolution(axis));
            end
            fsp_name = fullfile(obj.save_directory, "temp.fsp");
            appevalscript(obj.lumerical_session, sprintf('save("%s");',fsp_name));
            % add material
            appputvar(obj.lumerical_session, 'RI', obj.RI);
            for axis = 1:3
                appputvar(obj.lumerical_session, axis_name_list{axis}, (1:size(obj.RI, axis))*resolution(axis));
            end
            appevalscript(obj.lumerical_session, 'addimport;importnk2(RI, x, y, z);');
            for axis = 1:3
                target_name = axis_name_list{axis};
                if obj.PML_boundary(axis)
                    min_max_code = strcat(sprintf('min_%s = get("%s min");',target_name, target_name), ...
                                          sprintf('max_%s = get("%s max");',target_name, target_name));
                    num_code = sprintf('num_%s = get("data %s points");',target_name, target_name);
                else
                    min_max_code = strcat(sprintf('min_%s = (get("%s min")-d%s)*2/3 + get("%s max")*1/3+d%s;', target_name, target_name, target_name, target_name, target_name), ...
                                          sprintf('max_%s = (get("%s min")-d%s)*1/3 + get("%s max")*2/3;', target_name, target_name, target_name, target_name));
                    num_code = sprintf('num_%s = get("data %s points")/3;',target_name, target_name);
                end
                appevalscript(obj.lumerical_session,strcat(min_max_code, num_code));
            end
            % set boundary
            appevalscript(obj.lumerical_session, 'addfdtd;set("dimension", 2);set("min mesh step", 0);set("mesh type", 3);');
            for axis = 1:3
                target_name = axis_name_list{axis};
                boundary_size_code = sprintf('set("%s min", min_%s);set("%s max", max_%s);',target_name,target_name,target_name,target_name);
                if obj.PML_boundary(axis)
                    bc_code = sprintf('set("%s min bc", "PML");set("%s max bc", "PML");set("pml profile", 3);',target_name,target_name);
                else
                    bc_code = sprintf('set("%s min bc", "Bloch");',target_name);
                end
                mesh_code = sprintf('set("define %s mesh by",4);set("mesh cells %s", num_%s-1);',target_name, target_name, target_name);
                appevalscript(obj.lumerical_session, strcat(boundary_size_code, bc_code, mesh_code));
            end
            % add source
            k_0 =2*pi/current_source.wavelength;
            phi = asin(sqrt(sum(current_source.horizontal_k_vector .^2))/k_0);
            if phi == 0
                theta = 0;
            else
                theta = atan(current_source.horizontal_k_vector(2)/current_source.horizontal_k_vector(1));
            end
            rot_matrix = rotz(theta) * roty(phi);
            pol = circshift(current_source.polarization,-current_source.direction);
            horizontal_axis = rem([3 1] + current_source.direction, 3) + 1;
            horizontal_name1 = axis_name_list{horizontal_axis(1)};
            horizontal_name2 = axis_name_list{horizontal_axis(2)};
            for pol_idx = 0:1
                % pol_idx == 0 => p-pol, pol_idx == 1 => s-pol 
                pol_direction = rot_matrix * circshift([1 0 0], pol_idx)';
                complex_amp = dot(pol_direction, pol);
                if abs(complex_amp) ~= 0
                    source_insertion_code = strcat('addplane;', ...
                                                   sprintf('set("name", "pol%d");',pol_idx+1), ...
                                                   sprintf('set("center wavelength", %g);', obj.wavelength*microns), ...
                                                   sprintf('set("wavelength span", 0);'), ...
                                                   sprintf('set("polarization angle", %d);', 90*pol_idx), ...
                                                   sprintf('set("amplitude", %g);', abs(complex_amp)), ...
                                                   sprintf('set("phase", %g);', 180/pi*angle(complex_amp)), ...
                                                   sprintf('set("angle theta", %g);', 180/pi*theta), ...
                                                   sprintf('set("angle phi", %g);', 180/pi*phi), ...
                                                   sprintf('set("%s min", min_%s); set("%s max", max_%s);',horizontal_name1,horizontal_name1,horizontal_name1,horizontal_name1), ...
                                                   sprintf('set("%s min", min_%s); set("%s max", max_%s);',horizontal_name2,horizontal_name2,horizontal_name2,horizontal_name2), ...
                                                   sprintf('set("%s", min_%s);',axis_name_list{current_source.direction},axis_name_list{current_source.direction}) ...
                                                   );
                    appevalscript(obj.lumerical_session, source_insertion_code);
                end
            end
            % set profiler
            appevalscript(obj.lumerical_session, 'addpower;set("name","field_profile_vol");set("monitor type",8);');
            for axis = 1:3
                target_name = axis_name_list{axis};
                profiler_size_code = sprintf('set("%s min",min_%s+%g);set("%s max",min_%s+%g);',target_name,target_name,lumerical_roi(2*axis-1),target_name,target_name,lumerical_roi(2*axis));
                appevalscript(obj.lumerical_session, profiler_size_code);
            end
            % run simulation
            run_code = strcat('t_start = now;', ...
                               'run;', ...
                               't_code = now - t_start;');
            appevalscript(obj.lumerical_session, run_code);
            % get simulation results
            for field_name = ['E','H']
                appevalscript(obj.lumerical_session, strcat(sprintf('res_vol = getresult("field_profile_vol", "%s");',field_name), ...
                                                            sprintf('%sfield=reshape(res_vol.%s,[length(res_vol.x),length(res_vol.y),length(res_vol.z),3]);',field_name,field_name)));
            end
            Efield = appgetvar(obj.lumerical_session,'Efield');
            Hfield = appgetvar(obj.lumerical_session,'Hfield');
        end
    end
end