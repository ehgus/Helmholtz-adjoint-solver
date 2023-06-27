function [Efield,Hfield]=solve(obj, current_source)
    % generate incident field
    current_source.RI_bg = obj.RI_bg;
    incident_Efield = current_source.generate_Efield([obj.boundary_thickness_pixel; obj.boundary_thickness_pixel]);
    if obj.use_GPU
        incident_Efield = gpuArray(incident_Efield);
    end
    % evaluate scattered field
    [Efield, Hfield] = eval_scattered_field(obj,incident_Efield);
    % evaluate full field
    Efield = Efield + current_source.generate_Efield(zeros(2,3));
    Hfield = Hfield + current_source.generate_Hfield(zeros(2,3));
end
