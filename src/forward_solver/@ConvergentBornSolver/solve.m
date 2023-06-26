function [Efield,Hfield]=solve(obj, input_field)
    if ~obj.use_GPU
        input_field=single(input_field);
    else
        input_field=single(gpuArray(input_field));
    end

    assert(ndims(input_field) == 3 && size(input_field,3) == 2, 'The input field should have size of (xsize, ysize, 2)')

    input_field=fft2(input_field);
    %2D to 3D field
    input_field = ifftshift2(obj.transform_field_3D(fftshift2(input_field)));
    %compute
    [Efield, Hfield] = obj.solve_forward(input_field);
    %gather to release gpu memory
    obj.V=gather(obj.V);
    obj.field_attenuation_mask=gather(obj.field_attenuation_mask);
    obj.phase_ramp=gather(obj.phase_ramp);
end
