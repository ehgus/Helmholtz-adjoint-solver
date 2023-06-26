classdef AdjointSolver < OpticalSimulation
    properties
        % foward solver
        forward_solver;
        
        % FoM, weight and RoI
        mode = "Intensity"; % Transmission
        
        % optimization option
        optimizer = FistaOptim;
        itter_max = 100;
        nmin = -inf;
        nmax = inf;
        ROI_change;
        density_map;
        step = 0.1;
        tv_param = 1e-3;
        steepness = 0.5;
        RI_inter; % no need to be initialized

        % regularization
        use_non_negativity = false;

        % lithography uncertainty & resolution
        binarization_step = 100;
        spatial_diameter = 1;
        spatial_filter_range = [1 Inf];
        spatial_filter;  % no need to be initialized
        averaging_filter = [false false false]; 

        % used for solver
        gradient_full;
        gradient;
    end
    methods
        function obj = AdjointSolver(options)
            obj@OpticalSimulation(options);
            pixel_size = fix(obj.spatial_diameter./obj.forward_solver.resolution);
            % gaussian spatial filtering
            x_idx = linspace(-obj.spatial_diameter,obj.spatial_diameter,pixel_size(1));
            y_idx = linspace(-obj.spatial_diameter,obj.spatial_diameter,pixel_size(2));
            x_idx = reshape(x_idx,[],1); 
            y_idx = reshape(y_idx,1,[]); 
            obj.spatial_filter = exp(-(x_idx.^2+y_idx.^2)/(2*obj.spatial_diameter^2));
            if length(obj.spatial_filter) > 1
                obj.spatial_filter = obj.spatial_filter/sum(obj.spatial_filter,'all');
            else
                obj.spatial_filter = 1;
                warning('The spatial_diameter should be more larger than the resolution');
            end
        end

        function get_gradient(obj, E_adj, E_old, RI, index)
            % 3D gradient
            obj.gradient_full(:) = E_adj.*E_old;
            isRItensor = size(RI,4) == 3;
            if isRItensor
                obj.gradient(:) = sum(obj.gradient_full,5);
            else
                obj.gradient(:) = sum(obj.gradient_full,4:5);
            end

            % 2D gradient for lithography mask design
            obj.gradient(~obj.ROI_change) = nan;
            avg_dim = 1:3;
            avg_dim = avg_dim(obj.averaging_filter);
            if ~isempty(avg_dim)
                mean_gradient = mean(obj.gradient,avg_dim,'omitnan');
                obj.gradient(:) = 0;
                obj.gradient = obj.gradient + mean_gradient;
            end
            obj.gradient(~obj.ROI_change) = 0;
            % (RI + RI_gradient)^2 ~ V + gradient => RI_gradient = gradient/(2RI)
            obj.gradient(:) = obj.gradient./RI;
            % To project the RI_gradient on density,
            % RI_porjected_gradient = inner_product(RI_gradient,unit_RI_vector)*unit_RI_vector
            obj.density_map(:) = real(obj.nmax-obj.nmin)*real(obj.gradient)+imag(obj.nmax-obj.nmin)*imag(obj.gradient);
            % adaptive spatial filtering
            if obj.spatial_filter_range(1) < index && index < obj.spatial_filter_range(2)
                obj.density_map(:) = cconv2(obj.density_map,obj.spatial_filter);
                obj.gradient(obj.ROI_change) = obj.density_map(obj.ROI_change);
            end
            obj.gradient(:) = 1/2*(obj.nmax-obj.nmin)/abs(obj.nmax-obj.nmin)^2*obj.gradient;
        end

        function RI_opt = post_regularization(obj, RI_opt, index)
            obj.density_map(obj.ROI_change) = real(RI_opt(obj.ROI_change)-obj.nmin)/real(obj.nmax-obj.nmin); 
            % Min, Max regularization
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;

            % adaptive binarization
            beta = obj.steepness*fix(index/obj.binarization_step);
            if beta > obj.steepness/2
                tanh_value = tanh(beta/2);
                ROI_density_map = obj.density_map(obj.ROI_change);
                ROI_density_map(:) = (tanh_value + tanh(beta*(ROI_density_map-0.5)))/ ...
                                (tanh_value + tanh(beta/2));
                obj.density_map(obj.ROI_change) = ROI_density_map;
            end
            % update
            RI_opt(obj.ROI_change) = obj.density_map(obj.ROI_change)*(obj.nmax-obj.nmin)+obj.nmin;
        end
        function options = preprocess_params(obj, options)
            if obj.mode == "Transmission"
                % required: bunch of EM wave profiles on a plane, ROI
                % phase is not important
                assert(isfield(options, 'surface_vector'));
                assert(isfield(options, 'target_transmission'));
                assert(isfield(options, 'E_field')); % {x,y,z,direction}
                assert(isfield(options, 'H_field')); % {x,y,z,direction}: B_field is consistant with E_field
                options.normal_transmission = cell(1,length(options.E_field));
                for idx = 1:length(options.E_field)
                    normal_S = 2 * real(poynting_vector(options.E_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:), ...
                                                        options.H_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)));
                    options.normal_transmission{idx} = abs(sum(normal_S(:,:,end,3),1:2));
                end
            end
        end
    end
end