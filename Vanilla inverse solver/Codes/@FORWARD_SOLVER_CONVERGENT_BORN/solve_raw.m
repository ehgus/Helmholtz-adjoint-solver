function Field=solve_raw(h,incident_field)
    if (h.parameters.use_GPU)
        h.V=gpuArray(h.V);
        h.refocusing_util=gpuArray(h.refocusing_util);
        h.phase_ramp=gpuArray(h.phase_ramp);
        incident_field=gpuArray(single(incident_field));
        h.attenuation_mask=gpuArray(h.attenuation_mask);
        h.Greenp = gpuArray(h.Greenp);
        h.flip_Greenp = gpuArray(h.flip_Greenp);
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
    Field = Field + incident_field;

    if h.parameters.verbose
        set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]), colorbar, axis off,drawnow
        colormap hot
    end
end   