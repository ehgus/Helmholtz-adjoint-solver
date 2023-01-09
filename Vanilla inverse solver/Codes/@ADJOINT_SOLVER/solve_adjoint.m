function Field=solve_adjoint(h,incident_field, intensity_mask)
    if (h.forward_solver.use_GPU)
        incident_field=gpuArray(single(incident_field));
    end

    incident_field = incident_field.*intensity_mask;
    
    % Evaluate output field
    Field = h.forward_solver.eval_scattered_field(incident_field);
    Field = Field + incident_field;
end   