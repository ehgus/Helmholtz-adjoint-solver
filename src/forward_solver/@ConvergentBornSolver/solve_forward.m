function Field=solve_forward(h,incident_field)
    if (h.use_GPU)
        h.refocusing_util=gpuArray(h.refocusing_util);
        incident_field=gpuArray(single(incident_field));
    end
    
    % generate incident field
    incident_field = fftshift(ifft2(ifftshift(incident_field)));
    incident_field = h.padd_field2conv(incident_field);
    incident_field = fft2(ifftshift(incident_field));
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*h.refocusing_util;
    incident_field = fftshift(ifft2(incident_field));
    incident_field = h.crop_conv2RI(incident_field);

    h.refocusing_util=gather(h.refocusing_util);
    
    % Evaluate output field
    Field = eval_scattered_field(h,incident_field);
    Field = Field + gather(incident_field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:));
end