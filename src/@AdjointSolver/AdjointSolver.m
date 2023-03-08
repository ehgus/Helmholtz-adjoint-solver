classdef AdjointSolver < OpticalSimulation
    properties
        % foward solver
        forward_solver;
        
        % FoM, weight and RoI
        mode = "Intensity"; % Transmission
        
        % optimization option
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
        spatial_filtering_count = 3;
        spatial_filter;  % no need to be initialized
        averaging_filter = [false false false]; 

        % used for solver
        gradient_full;
        gradient;
    end
    methods
        function h=AdjointSolver(options)
            h@OpticalSimulation(options);
            assert(strcmp(h.mode, "Intensity"),"Transmission mode is not implemented yet")
            spatial_radius = h.spatial_diameter/2;
            pixel_size = fix(h.spatial_diameter./h.forward_solver.resolution);
            x_idx = reshape(linspace(-spatial_radius,spatial_radius,pixel_size(1)),[],1);
            y_idx = reshape(linspace(-spatial_radius,spatial_radius,pixel_size(2)),1,[]);
            h.spatial_filter = spatial_radius - sqrt(x_idx.^2+y_idx.^2);
            h.spatial_filter(h.spatial_filter < 0) = 0;
            assert(sum(h.spatial_filter,'all') >0, 'The spatial_diameter should be more larger');
            h.spatial_filter = h.spatial_filter/sum(h.spatial_filter,'all');
        end

        function get_gradient(h,E_adj,E_old,isRItensor)
            % 3D gradient
            h.gradient_full(:) = E_adj.*E_old;
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
        end

        function RI_opt = update_gradient(h,RI_opt,RI,step_size)
            % update gradient based on the density-based 
            % (RI + RI_gradient)^2 ~ V + gradient => RI_gradient = gradient/(2RI)
            % TO project the RI_gradient on density,
            % RI_porjected_gradient = inner_product(RI_gradient,unit_RI_vector)*unit_RI_vector
            h.gradient(:) = h.gradient./RI;
            h.gradient(:) = real(h.nmax-h.nmin)*real(h.gradient)+imag(h.nmax-h.nmin)*imag(h.gradient);
            h.gradient(:) = step_size/2*(h.nmax-h.nmin)/abs(h.nmax-h.nmin)^2*h.gradient;
            RI_opt(:) = RI + h.gradient;
        end

        function RI_opt = post_regularization(h,RI_opt,index)
            h.density_map(:) = real(RI_opt-h.nmin)/real(h.nmax-h.nmin); 
            % Min, Max regularization
            h.density_map(h.density_map > 1) = 1;
            h.density_map(h.density_map < 0) = 0;
            % adaptive spatial filtering
            if index +h.spatial_filtering_count > h.itter_max
                h.density_map(:) = cconv2(h.density_map,h.spatial_filter);
            end
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
    end
end