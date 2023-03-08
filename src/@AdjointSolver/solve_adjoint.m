function [Field, FoM] =solve_adjoint(h,incident_field, options)
    % Generate adjoint source and calculate adjoint field
    if (h.forward_solver.use_GPU)
        incident_field=gpuArray(single(incident_field));
    end

    if h.mode == "Intensity"
        incident_field = incident_field.*options.intensity_weight;
        FoM = sum(abs(incident_field).^2.*options.intensity_weight,'all') / numel(incident_field);
        incident_field = padarray(incident_field,h.forward_solver.boundary_thickness_pixel,0);
    elseif h.mode == "Transmission"
        % Calculate transmission rate of plane wave
        error("Not implemented")
    end
    % Evaluate output field
    Field = h.forward_solver.eval_scattered_field(incident_field);
    Field = Field + gather(incident_field(h.forward_solver.ROI(1):h.forward_solver.ROI(2),h.forward_solver.ROI(3):h.forward_solver.ROI(4),h.forward_solver.ROI(5):h.forward_solver.ROI(6),:));
end   