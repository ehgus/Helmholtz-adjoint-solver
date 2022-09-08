classdef TOPOLOGY_OPTIMIZER < STRUCT_CLASS
    properties
        %% solver used
        forward_solver;
        adjoint_solver;
        
        %% Iteration parameters
        % basic updater
        step = 0.1;
        max_iteration = 150;
        ROI_change = [];

        % non-negativity
        use_non_negativity = false;
        density_thresholding = false;
        RImin = -inf;
        RImax = inf;
    end
    methods
        function h = TOPOLOGY_OPTIMIZER(struct_parameters)
            h = update_properties(h, struct_parameters);
            if isempty(h.forward_solver) || isempty(h.adjoint_solver)
                error('Forward and backward solver should be specified');
            end 
        end

        function RI_opt = optimize_RI(h, input_field, intensity_mask, RI)
            % Iteratively update the RI to maximize 
            assert(mod(size(RI,3),2)==1, 'Length of RI block along z axis should be odd');

            % set parameters
            RI_opt=single(RI);
            if h.verbose
                Figure_of_Merit=zeros(h.parameters.itter_max,1);
            end
            
            t_n=0;
            t_np=1;

            s_n=RI_opt.^2;
            u_n=RI_opt.^2;
            x_n=RI_opt.^2;

            % main
            for i = 1:h.max_iteration
                tic;
                % set RI map to be optimized
                h.forward_solver.set_RI(RI_opt);
                h.adjoint_solver.set_RI(RI_opt);
                
                % calculate gradient & regularization
                [~,~,E_field] = h.forward_solver.solve(input_field);
                E_adj=h.adjoint_solver.solve_adjoint(E_field, intensity_mask);
                %gradient = h.get_gradient(E_adj,E_field);
                gradient = real(E_adj.*E_field);
                gradient = sum(gradient,4);
                gradient(~h.ROI_change) = 0;
                clear E_field
                clear E_adj
                % update
                t_n = t_np;
                t_np = (1+sqrt(1+4*t_n^2))/2;
                
                s_n = u_n-h.step*gradient;
                clear gradient
                % post regularization
                RI_opt=sqrt(s_n);
                if h.use_non_negativity
                    RI_opt((h.ROI_change(:)).*(real(RI_opt(:)) > real(h.RImax))==1) = h.RImax;
                    RI_opt((h.ROI_change(:)).*(real(RI_opt(:)) < real(h.RImin))==1) = h.RImin;
                end
                s_n = RI_opt.^2;

                u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                x_n=s_n;
                % verbose
                if h.verbose
                    Figure_of_Merit(i) = sum(abs(E_field).^2.*intensity_mask,'all')/sum(abs(E_old).^2,'all');
                end
                exec_time = toc;
                fprintf('processing %d of %d (%.5f s) \n',i, h.max_iteration, exec_time);
            end
            RI_opt=sqrt(u_n);
        end

        function gradient = get_gradient(h,E_adj,E_field)
            gradient = real(E_adj.*E_field);
            gradient = sum(gradient,4);
            gradient(~h.ROI_change) = 0;
        end
    end
end