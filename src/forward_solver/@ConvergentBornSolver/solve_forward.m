function [Field, Hfield] = solve_forward(obj,incident_field)
    if (obj.use_GPU)
        obj.refocusing_util=gpuArray(obj.refocusing_util);
        incident_field=gpuArray(single(incident_field));
    end
    
    % generate incident field
    incident_field = fftshift(ifft2(ifftshift(incident_field)));
    incident_field = obj.padd_field2conv(incident_field);
    incident_field = fft2(ifftshift(incident_field));
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*obj.refocusing_util;
    incident_field = fftshift(ifft2(incident_field));
    incident_field = obj.crop_conv2RI(incident_field);
    if obj.vector_simulation
        incident_field_h = -1i * (obj.wavelength/2/pi)/(120*pi) * obj.curl_field(incident_field);
    end

    obj.refocusing_util=gather(obj.refocusing_util);
    
    % Evaluate output field
    [Field, Hfield] = eval_scattered_field(obj,incident_field);
    Field = Field + gather(incident_field(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:));
    if obj.vector_simulation
        Hfield = Hfield + gather(incident_field_h(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:));
    end
end