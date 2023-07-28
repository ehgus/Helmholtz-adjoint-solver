classdef AdjointSolver < OpticalSimulation
    properties
        forward_solver
        % optimization option
        optim_mode   {mustBeMember(optim_mode,["Intensity","Transmission"])} = "Intensity"
        max_iter     {mustBePositive, mustBeInteger} = 100
        optimizer
    end
    properties(Hidden)
        gradient
    end
    methods
        function obj = AdjointSolver(options)
            obj@OpticalSimulation(options);
        end

        function RI_opt=solve(obj, current_source, RI, options)
            % initialize parameters
            RI_opt=single(RI);
            figure_of_merit=NaN(1,obj.max_iter);
            obj.optimizer.reset();
            obj.gradient = complex(zeros(size(RI,1:4),'single'));
            options = obj.preprocess_params(options);
            RI_opt = obj.optimizer.preprocess(RI_opt);
            % main
            for iter_idx = 1:obj.max_iter
                fprintf('Iteration: %d\n', iter_idx);
                t_start = tic;
                % Calculated gradient RI based on intensity mode
                obj.forward_solver.set_RI(RI_opt);
                [E_fwd, H_fwd] = obj.forward_solver.solve(current_source);
                [E_adj, figure_of_merit(iter_idx)]=obj.solve_adjoint(E_fwd, H_fwd, options);
                obj.get_gradient(E_adj, E_fwd, RI_opt);
                RI_opt = obj.optimizer.apply_gradient(RI_opt, obj.gradient, iter_idx);
                t_end = toc(t_start);
                if obj.verbose
                    figure(1)
                    line(1:iter_idx, figure_of_merit(1:iter_idx))
                    figure(2)
                    imagesc(real(RI_opt(:,:,8)))
                    fprintf('Elapsed time is %.6f seconds\n',t_end)
                    drawnow;
                end
            end
            RI_opt = obj.optimizer.postprocess(RI_opt);
            RI_opt=gather(RI_opt);
        end

        function options = preprocess_params(obj, options)
            if obj.optim_mode == "Transmission"
                % required: bunch of EM wave profiles on a plane, ROI
                % phase is not important
                assert(isfield(options, 'surface_vector'));
                assert(isfield(options, 'target_transmission'));
                assert(isfield(options, 'E_field')); % {x,y,z,direction}
                assert(isfield(options, 'H_field')); % {x,y,z,direction}: H_field is consistant with E_field
                options.normal_transmission = cell(1,length(options.E_field));
                for idx = 1:length(options.E_field)
                    normal_S = 2 * real(poynting_vector(options.E_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:), ...
                                                        options.H_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)));
                    options.normal_transmission{idx} = abs(sum(normal_S(:,:,end,3),1:2));
                end
            end
        end

        function get_gradient(obj, E_adj, E_old, RI)
            % 3D gradient
            isRItensor = size(RI,4) == 3;
            obj.gradient(:) = 0;
            if isRItensor
                obj.gradient(:) = E_adj.*E_old;
            else
                for axis = 1:3
                    obj.gradient(:) = obj.gradient + E_adj(:,:,:,axis).*E_old(:,:,:,axis);
                end
            end
            % (RI + RI_gradient)^2 ~ V + gradient => RI_gradient = gradient/(2RI)
            obj.gradient(:) = obj.gradient./RI;
        end
    end
end