function [Field, FoM] =solve_adjoint(obj, E_fwd, H_fwd, options)
    % minimize FoM
    % Generate adjoint source and calculate adjoint field
    if (obj.forward_solver.use_GPU)
        E_fwd = gpuArray(E_fwd);
    end
    % pre-allocate empty array
    if obj.optim_mode == "Intensity"
        grid_size = obj.forward_solver.size + 2 * obj.forward_solver.boundary_thickness_pixel;
        grid_size(4) = 3;
        if obj.forward_solver.use_GPU
            adjoint_field = zeros(grid_size,'single','gpuArray');
        else
            adjoint_field = zeros(grid_size,'single');
        end
        FoM = - sum(abs(E_fwd).^2.*options.intensity_weight,'all') / numel(E_fwd);
        adjoint_field(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:) = -conj(E_fwd).*options.intensity_weight;
        obj.forward_solver.set_RI(obj.forward_solver.RI);
        Field = obj.forward_solver.eval_scattered_field(adjoint_field);
    elseif obj.optim_mode == "Transmission"
        % Calculate transmission rate of plane wave
        % relative intensity: Matrix of relative intensity
        relative_transmission = zeros(1,length(options.E_field));
        for idx = 1:length(options.E_field)
            eigen_S = poynting_vector(E_fwd, options.H_field{idx}) + poynting_vector(conj(options.E_field{idx}), conj(H_fwd));
            relative_transmission(idx) = sum(eigen_S(:,:,end,3),'all');
        end
        relative_transmission = relative_transmission./options.normal_transmission;
        adjoint_source_weight = abs(relative_transmission).^2 - options.target_transmission;
        for idx = 1:length(relative_transmission)
            val = relative_transmission(idx);
            if isfinite(val/abs(val))
                adjoint_source_weight(idx) = adjoint_source_weight(idx)*(val/abs(val));
            end
        end
        adj_current_source = options.current_source(idx);
        for idx = 1:length(options.current_source)
            src = PlaneSource(options.current_source(idx));
            src.polarization = src.polarization.*(-1i*adjoint_source_weight(idx));
            src.outcoming_wave = ~src.outcoming_wave;
            src.polarization = conj(src.polarization);
            adj_current_source(idx) = src;
        end
        % figure of merit
        FoM = sum(abs(adjoint_source_weight).^2,'all');
        options.forward_solver.set_RI(obj.forward_solver.RI);
        Field = options.forward_solver.solve(adj_current_source);
    end
end
