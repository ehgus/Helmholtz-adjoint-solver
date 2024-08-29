classdef FDTDsolver < ForwardSolver
    properties
        % FDTD option
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
            obj.RI= double(RI);
        end
        function [Efield, Hfield]=solve(obj,current_source)
            assert(isa(current_source, 'PlaneSource'), "PlaneSource is the only available source")
            addpath(obj.lumerical_api_dir);
            axis_name_list = {'x', 'y', 'z'};
            microns = 1e-6;
            resolution = obj.resolution .* microns;
            lumerical_roi = [
                1 size(obj.RI,1), ...
                1 size(obj.RI,2), ...
                1 size(obj.RI,3) ...
            ];
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
            fsp_name = fullfile(obj.save_directory, "Temp.fsp");
            appevalscript(obj.lumerical_session, sprintf('save("%s");',fsp_name));
            % add material
            RI_padded = obj.RI;
            for idx = 1:3
                if obj.PML_boundary(idx)
                    RI_padded = padarray(RI_padded, circshift([0,0,1],idx), "replicate");
                else
                    RI_padded = padarray(RI_padded, circshift([0,0,1],idx), "circular");
                end
            end
            appputvar(obj.lumerical_session, 'RI', RI_padded);
            for axis = 1:3
                appputvar(obj.lumerical_session, axis_name_list{axis}, (0:size(RI_padded, axis)-1)*resolution(axis));
            end
            appevalscript(obj.lumerical_session, 'addimport;importnk2(RI, x, y, z);');
            for axis = 1:3
                target_name = axis_name_list{axis};
                min_max_code = strcat(sprintf('min_%s = get("%s min")+d%s/2;',target_name, target_name, target_name), ...
                                      sprintf('max_%s = get("%s max")-d%s/2;',target_name, target_name, target_name));
                appevalscript(obj.lumerical_session,min_max_code);
            end
            % set boundary
            appevalscript(obj.lumerical_session, 'addfdtd;set("dimension", 2);set("min mesh step", 0);set("mesh type", 3);');
            for axis = 1:3
                target_name = axis_name_list{axis};
                boundary_size_code = sprintf('set("%s min", min_%s);set("%s max", max_%s);',target_name,target_name,target_name,target_name);
                if obj.PML_boundary(axis)
                    bc_code = sprintf('set("%s min bc", "PML");set("%s max bc", "PML");set("pml profile", 3);',target_name,target_name);
                else
                    bc_code = sprintf('set("%s min bc", "Periodic");',target_name);
                end
                mesh_code = sprintf('set("define %s mesh by","maximum mesh step");set("d%s", d%s);',target_name, target_name, target_name);
                appevalscript(obj.lumerical_session, strcat(boundary_size_code, bc_code, mesh_code));
            end
            % add source
            for src_idx = 1:length(current_source)
                src = current_source(src_idx);
                src.RI_bg = obj.RI_bg;
                k_0 =2*pi*src.RI_bg/src.wavelength;
                theta = asind(sqrt(sum(src.horizontal_k_vector .^2))/k_0);
                if theta == 0
                    phi = 0;
                else
                    phi = atand(src.horizontal_k_vector(2)/src.horizontal_k_vector(1));
                end
                rot_matrix = rotz(phi) * roty(theta);
                pol = circshift(src.polarization,-src.direction);
                horizontal_axis = rem([3 1] + src.direction, 3) + 1;
                center_xyz = src.center_position.*resolution;
                horizontal_name1 = axis_name_list{horizontal_axis(1)};
                horizontal_name2 = axis_name_list{horizontal_axis(2)};
                for pol_idx = 0:1
                    % pol_idx == 0 => p-pol, pol_idx == 1 => s-pol 
                    pol_direction = rot_matrix * circshift([1 0 0], pol_idx)';
                    complex_amp = dot(pol_direction, pol);
                    if abs(complex_amp) ~= 0
                        horizontal_center1 = center_xyz(horizontal_name1 - 'x' + 1);
                        horizontal_center2 = center_xyz(horizontal_name2 - 'x' + 1);
                        source_insertion_code = strcat('addplane;', ...
                                                    sprintf('set("name", "src%d_pol%d_forward");',src_idx,pol_idx+1), ...
                                                    sprintf('set("injection axis",%d);',src.direction), ...
                                                    sprintf('set("center wavelength", %g);', obj.wavelength*microns), ...
                                                    sprintf('set("wavelength span", 0);'), ...
                                                    sprintf('set("polarization angle", %d);', 90*pol_idx), ...
                                                    sprintf('set("amplitude", %g);', abs(complex_amp)), ...
                                                    sprintf('set("phase", %g);', 180/pi*angle(complex_amp)), ...
                                                    sprintf('set("angle theta", %g);', theta), ...
                                                    sprintf('set("angle phi", %g);', phi), ...
                                                    sprintf('set("%s min", min_%s - max([0,min_%s + max_%s - %g])); set("%s max", max_%s - min([0,min_%s + max_%s - %g]));', ...
                                                              horizontal_name1,horizontal_name1,horizontal_name1,horizontal_name1,2*horizontal_center1,horizontal_name1,horizontal_name1,horizontal_name1,horizontal_name1,2*horizontal_center1), ...
                                                    sprintf('set("%s min", min_%s - max([0,min_%s + max_%s - %g])); set("%s max", max_%s - min([0,min_%s + max_%s - %g]));', ...
                                                              horizontal_name2,horizontal_name2,horizontal_name2,horizontal_name2,2*horizontal_center2,horizontal_name2,horizontal_name2,horizontal_name2,horizontal_name2,2*horizontal_center2), ...
                                                    sprintf('set("%s", min_%s);',axis_name_list{src.direction},axis_name_list{src.direction}) ...
                                                    );
                        appevalscript(obj.lumerical_session, source_insertion_code);
                        source_insertion_code_opposite = strcat('addplane;', ...
                                                    sprintf('set("name", "src%d_pol%d_backward");',src_idx,pol_idx+1), ...
                                                    sprintf('set("direction","Backward");'), ...
                                                    sprintf('set("injection axis",%d);',src.direction), ...
                                                    sprintf('set("center wavelength", %g);', obj.wavelength*microns), ...
                                                    sprintf('set("wavelength span", 0);'), ...
                                                    sprintf('set("polarization angle", %d);', 90*pol_idx), ...
                                                    sprintf('set("amplitude", %g);', abs(complex_amp)), ...
                                                    sprintf('set("phase", %g);', 180/pi*angle(complex_amp)), ...
                                                    sprintf('set("angle theta", %g);', -theta), ...
                                                    sprintf('set("angle phi", %g);', phi), ...
                                                    sprintf('set("%s min", min_%s - max([0,min_%s + max_%s - %g])); set("%s max", max_%s - min([0,min_%s + max_%s - %g]));', ...
                                                              horizontal_name1,horizontal_name1,horizontal_name1,horizontal_name1,2*horizontal_center1,horizontal_name1,horizontal_name1,horizontal_name1,horizontal_name1,2*horizontal_center1), ...
                                                    sprintf('set("%s min", min_%s - max([0,min_%s + max_%s - %g])); set("%s max", max_%s - min([0,min_%s + max_%s - %g]));', ...
                                                              horizontal_name2,horizontal_name2,horizontal_name2,horizontal_name2,2*horizontal_center2,horizontal_name2,horizontal_name2,horizontal_name2,horizontal_name2,2*horizontal_center2), ...
                                                    sprintf('set("%s", min_%s);',axis_name_list{src.direction},axis_name_list{src.direction}) ...
                                                    );
                        appevalscript(obj.lumerical_session, source_insertion_code_opposite);
                    end
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