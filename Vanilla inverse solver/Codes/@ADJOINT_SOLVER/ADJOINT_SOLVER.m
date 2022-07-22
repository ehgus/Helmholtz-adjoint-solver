classdef ADJOINT_SOLVER
    properties %(SetAccess = protected, Hidden = true)
        forward_solver;
        overlap_count;
        filter;
        RI_inter;
        parameters;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@BACKWARD_SOLVER();
            
            %specific parameters
            params.forward_solver= @(x) FORWARD_SOLVER(x);
            params.forward_solver_parameters= FORWARD_SOLVER.get_default_parameters();

            % Iteration parameters
            params.step=0.01;%0.01;0.01;%0.01;
%             params.tv_param=0.001;%0.1;
%             params.use_non_negativity=false;
%             params.nmin = -inf;%1.336;
%             params.nmax = inf;%1.6;
%             params.kappamax = 0; % imaginary RI
%             params.inner_itt = 100; % imaginary RI
%             params.itter_max = 100; % imaginary RI
%             params.num_scan_per_iteration = 0; % 0 -> every scan is used
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
        
        function h=ADJOINT_SOLVER(params)
            h.parameters = params;
            h.forward_solver = h.parameters.forward_solver(h.parameters.forward_solver_parameters);
        end

    end
end