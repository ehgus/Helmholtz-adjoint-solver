classdef ADJOINT_SOLVER < handle
    properties %(SetAccess = protected, Hidden = true)
        forward_solver;
        overlap_count;
        filter;
        RI_inter;
        parameters;
        nmin;
        nmax;

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
        end

        function get_gradeint(h,E_adj,E_old,isRItensor)
            h.gradient_full(:) = real(E_adj.*E_old);
            if isRItensor
                h.gradient(:) = sum(h.gradient_full,5);
            else
                h.gradient(:) = sum(h.gradient_full,4:5);
            end
            h.gradient(~h.parameters.ROI_change) = 0;
        end
    end
end