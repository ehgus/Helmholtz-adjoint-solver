function [Field, FoM] =solve_adjoint(obj, E_fwd, H_fwd, options)
    % minimize FoM
    % Generate adjoint source and calculate adjoint field
    if (obj.forward_solver.use_GPU)
        E_fwd = gpuArray(E_fwd);
    end
    % pre-allocate empty array
    Nsize = obj.forward_solver.size + 2 * obj.forward_solver.boundary_thickness_pixel;
    Nsize(4) = 3;
    if obj.forward_solver.use_GPU
        adjoint_field = zeros(Nsize,'single','gpuArray');
    else
        adjoint_field = zeros(Nsize,'single');
    end
    if obj.mode == "Intensity"
        FoM = - sum(abs(E_fwd).^2.*options.intensity_weight,'all') / numel(E_fwd);
        adjoint_field(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:) = -conj(E_fwd).*options.intensity_weight;
        Field = obj.forward_solver.eval_scattered_field(adjoint_field);
    elseif obj.mode == "Transmission"
        % Calculate transmission rate of plane wave
        % relative intensity: Matrix of relative intensity
        relative_transmission = zeros(1,length(options.E_field));
        for idx = 1:length(options.E_field)
            eigen_S = poynting_vector(E_fwd, options.H_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)) ...
                    + poynting_vector(conj(options.E_field{idx}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)), conj(H_fwd));
            relative_transmission(idx) = mean(sum(eigen_S(:,:,end,3),1:2)./options.normal_transmission{idx},'all');
        end
        adjoint_source_weight = (abs(relative_transmission).^2 - options.target_transmission).*(relative_transmission./abs(relative_transmission));
        disp(abs(relative_transmission).^2)
        for idx = 1:length(options.E_field)
            adjoint_field = adjoint_field - 1i * conj(options.E_field{idx}* adjoint_source_weight(idx));
        end
        
        % ad-hoc solution: It places input field at the edge of the simulation region.
        adjoint_field(:,:,obj.forward_solver.ROI(6)+1:end,:) = flip(adjoint_field(:,:,2*obj.forward_solver.ROI(6)-end+1:obj.forward_solver.ROI(6),:),3);

        % figure of merit
        FoM = mean(abs(adjoint_source_weight).^2,'all');
        options.forward_solver.set_RI(obj.forward_solver.RI);
        Field = options.forward_solver.eval_scattered_field(adjoint_field);
    end
    % Evaluate output field
    Field = Field + gather(adjoint_field(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:));
end
