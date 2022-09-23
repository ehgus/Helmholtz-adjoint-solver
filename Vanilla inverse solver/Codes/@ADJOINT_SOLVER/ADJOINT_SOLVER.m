classdef ADJOINT_SOLVER < handle
    properties %(SetAccess = protected, Hidden = true)
        forward_solver;
        overlap_count;
        filter;
        RI_inter;
        parameters;
        nmin;
        nmax;
        density_map;
        steepness;
        binarization_step;

        % used for solver
        gradient_full;
        gradient;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=BASIC_OPTICAL_PARAMETER();
            
            %specific parameters
            params.forward_solver= {};

            % Iteration parameters
            params.step=0.01;
            params.tv_param=0.001;
            params.use_non_negativity=false;
            params.nmin = -inf;
            params.nmax = inf;
            params.kappamax = 0; % imaginary RI
            params.inner_itt = 100; % imaginary RI
            params.itter_max = 100; % imaginary RI
            params.num_scan_per_iteration = 0; % 0 -> every scan is used
            params.steepness = 0.5;
            params.binarization_step = 100;
            params.verbose = true;
            % Adjoint mode parameters
            params.mode = "Intensity"; %"Transmission"
            params.ROI_change = [];
            %params.filter_by_count=false;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        Field=solve_adjoint(h,E_old,source)

        function h=ADJOINT_SOLVER(forward_solver, params)
            h.parameters = params;
            h.forward_solver = forward_solver;
            h.nmin = params.nmin;
            h.nmax = params.nmax;
            h.steepness = params.steepness; % it should be nonnegative value
            h.binarization_step = params.binarization_step;
        end

        function get_gradient(h,E_adj,E_old,isRItensor)
            % 3D gradient
            h.gradient_full(:) = real(E_adj.*E_old);
            if isRItensor
                h.gradient(:) = sum(h.gradient_full,5);
            else
                h.gradient(:) = sum(h.gradient_full,4:5);
            end

            % 2D gradient for lithography mask design
            h.gradient(~h.parameters.ROI_change) = nan;
            mean_2D = mean(h.gradient,3,'omitnan');
            for i = 1:size(h.gradient,3)
                h.gradient(:,:,i) = mean_2D;
            end
            h.gradient(~h.parameters.ROI_change) = 0;
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
            % adaptive binarization
            steepness = h.steepness*fix(index/h.binarization_step);
            if steepness > h.steepness/2
                tanh_value = tanh(steepness/2);
                ROI_density_map = h.density_map(h.parameters.ROI_change);
                ROI_density_map(:) = (tanh_value + tanh(steepness*(ROI_density_map-0.5)))/ ...
                                (tanh_value + tanh(steepness/2));
                h.density_map(h.parameters.ROI_change) = ROI_density_map;
            end
            % update
            RI_opt(h.parameters.ROI_change) = h.density_map(h.parameters.ROI_change)*(h.nmax-h.nmin)+h.nmin;
        end
    end
end