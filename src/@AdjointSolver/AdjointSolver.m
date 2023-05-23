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
        function h=AdjointSolver(options)
            h@OpticalSimulation(options);
            pixel_size = fix(h.spatial_diameter./h.forward_solver.resolution);
            % gaussian spatial filtering
            x_idx = linspace(-h.spatial_diameter,h.spatial_diameter,pixel_size(1));
            y_idx = linspace(-h.spatial_diameter,h.spatial_diameter,pixel_size(2));
            x_idx = reshape(x_idx,[],1); 
            y_idx = reshape(y_idx,1,[]); 
            h.spatial_filter = exp(-(x_idx.^2+y_idx.^2)/(2*h.spatial_diameter^2));
            if length(h.spatial_filter) > 1
                h.spatial_filter = h.spatial_filter/sum(h.spatial_filter,'all');
            else
                h.spatial_filter = 1;
                warning('The spatial_diameter should be more larger than the resolution');
            end
        end

        function get_gradient(h, E_adj, E_old, RI, index)
            % 3D gradient
            h.gradient_full(:) = E_adj.*E_old;
            isRItensor = size(RI,4) == 3;
            if isRItensor
                h.gradient(:) = sum(h.gradient_full,5);
            else
                h.gradient(:) = sum(h.gradient_full,4:5);
            end

            % 2D gradient for lithography mask design
            h.gradient(~h.ROI_change) = nan;
            avg_dim = 1:3;
            avg_dim = avg_dim(h.averaging_filter);
            if ~isempty(avg_dim)
                mean_gradient = mean(h.gradient,avg_dim,'omitnan');
                h.gradient(:) = 0;
                h.gradient = h.gradient + mean_gradient;
            end
            h.gradient(~h.ROI_change) = 0;
            % (RI + RI_gradient)^2 ~ V + gradient => RI_gradient = gradient/(2RI)
            h.gradient(:) = h.gradient./RI;
            % To project the RI_gradient on density,
            % RI_porjected_gradient = inner_product(RI_gradient,unit_RI_vector)*unit_RI_vector
            h.density_map(:) = real(h.nmax-h.nmin)*real(h.gradient)+imag(h.nmax-h.nmin)*imag(h.gradient);
            % adaptive spatial filtering
            if h.spatial_filter_range(1) < index && index < h.spatial_filter_range(2)
                h.density_map(:) = cconv2(h.density_map,h.spatial_filter);
                h.gradient(h.ROI_change) = h.density_map(h.ROI_change);
            end
            h.gradient(:) = 1/2*(h.nmax-h.nmin)/abs(h.nmax-h.nmin)^2*h.gradient;
        end

        function RI_opt = post_regularization(h,RI_opt,index)
            h.density_map(h.ROI_change) = real(RI_opt(h.ROI_change)-h.nmin)/real(h.nmax-h.nmin); 
            % Min, Max regularization
            h.density_map(h.density_map > 1) = 1;
            h.density_map(h.density_map < 0) = 0;

            % adaptive binarization
            beta = h.steepness*fix(index/h.binarization_step);
            if beta > h.steepness/2
                tanh_value = tanh(beta/2);
                ROI_density_map = h.density_map(h.ROI_change);
                ROI_density_map(:) = (tanh_value + tanh(beta*(ROI_density_map-0.5)))/ ...
                                (tanh_value + tanh(beta/2));
                h.density_map(h.ROI_change) = ROI_density_map;
            end
            % update
            RI_opt(h.ROI_change) = h.density_map(h.ROI_change)*(h.nmax-h.nmin)+h.nmin;
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
                for i = 1:length(options.E_field)
                    normal_S = 2 * real(poynting_vector(options.E_field{i}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:), ...
                                                        options.H_field{i}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)));
                    options.normal_transmission{i} = abs(sum(normal_S(:,:,end-10:end,3),1:2));
                end
            end
        end
    end
end