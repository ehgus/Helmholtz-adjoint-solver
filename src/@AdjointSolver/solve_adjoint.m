function [Field, FoM] =solve_adjoint(obj,conjugate_Efield, conjugate_Hfield, options)
    % minimize FoM
    % Generate adjoint source and calculate adjoint field
    if (obj.forward_solver.use_GPU)
        conjugate_Efield = gpuArray(conjugate_Efield);
    end
    % pre-allocate empty array
    Nsize = obj.forward_solver.size + 2 * obj.forward_solver.boundary_thickness_pixel;
    if obj.forward_solver.vector_simulation
        Nsize(4) = 3;
    end
    if obj.forward_solver.use_GPU
        adjoint_field = zeros(Nsize,'single','gpuArray');
    else
        adjoint_field = zeros(Nsize,'single');
    end
    if obj.mode == "Intensity"
        FoM = - sum(abs(conjugate_Efield).^2.*options.intensity_weight,'all') / numel(conjugate_Efield);
        adjoint_field(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:) = conjugate_Efield.*options.intensity_weight;
    elseif obj.mode == "Transmission"
        % Calculate transmission rate of plane wave
        % relative intensity: Matrix of relative intensity
        relative_transmission = zeros(1,length(options.E_field));
        for i = 1:length(options.E_field)
            eigen_S = poynting_vector(conj(conjugate_Efield), options.H_field{i}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)) ...
                   +  poynting_vector(conj(options.E_field{i}(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:)), conjugate_Hfield);
            relative_transmission(i) = sum(eigen_S .* options.surface_vector,'all');
        end
        relative_transmission = relative_transmission./options.normal_transmission;
        adjoint_source_weight = (abs(relative_transmission).^2 - options.target_transmission).*(relative_transmission/abs(relative_transmission));
        for i = 1:length(options.E_field)
            adjoint_field = adjoint_field + conj(options.E_field{i})*1i* adjoint_source_weight(i);
        end
        % figure of merit
        FoM = -mean(abs(adjoint_source_weight).^2,'all');
    end
    % Evaluate output field
    Field = obj.forward_solver.eval_scattered_field(adjoint_field);
    Field = Field + gather(adjoint_field(obj.forward_solver.ROI(1):obj.forward_solver.ROI(2),obj.forward_solver.ROI(3):obj.forward_solver.ROI(4),obj.forward_solver.ROI(5):obj.forward_solver.ROI(6),:));
end
