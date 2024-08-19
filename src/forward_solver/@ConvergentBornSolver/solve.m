function [Efield,Hfield]=solve(obj, current_source)
    assert(~isempty(obj.RI), "RI should be set before")
    % generate source:VE0
    current_source.RI_bg = obj.RI_bg;
    incident_Efield = current_source.generate_Efield([obj.boundary_thickness_pixel; obj.boundary_thickness_pixel]);
    if obj.use_GPU
        incident_Efield = gpuArray(single(incident_Efield));
        obj.V = gpuArray(single(obj.V));
    end
    is_isotropic = size(obj.V, 4) == 1;
    if is_isotropic
        source = obj.V .* incident_Efield;
    else % tensor
        source = zeros('like',incident_Efield);
        for axis = 1:3
            source = source + obj.V(:,:,:,:,axis) .* incident_Efield(:,:,:,axis);
        end
    end
    source = source + 1i*obj.eps_imag * incident_Efield;
    % evaluate scattered field
    [Efield, Hfield] = eval_scattered_field(obj,source);
    % evaluate full field
    Efield = Efield + current_source.generate_Efield(zeros(2,3));
    Hfield = Hfield + current_source.generate_Hfield(zeros(2,3));
end
