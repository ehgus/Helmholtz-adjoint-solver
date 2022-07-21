function Field=solve_raw(h,source)
    if (h.parameters.use_GPU)
        h.V=gpuArray(h.V);
        h.refocusing_util=gpuArray(h.refocusing_util);
        h.phase_ramp=gpuArray(h.phase_ramp);
        source=gpuArray(single(source));
        h.attenuation_mask=gpuArray(h.attenuation_mask);
        h.rads = gpuArray(h.rads);
        h.eye_3 = gpuArray(h.eye_3);
        h.Greenp = gpuArray(h.Greenp);
    end
    
    % generate incident field
    source = fftshift(ifft2(ifftshift(source)));
    source = h.padd_field2conv(source);
    source = fftshift(fft2(ifftshift(source)));
    source = ((reshape(source, [size(source,1),size(source,2),1,size(source,3)])).*h.refocusing_util);
    source = fftshift(ifft2(ifftshift(source)));
    source = h.crop_conv2RI(source);

    h.refocusing_util=gather(h.refocusing_util);
    
    % Evaluate output field
    incident_field = source;
    Field = eval_scattered_field(h,source);
    Field = Field + incident_field;

    if h.parameters.verbose
        set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]), colorbar, axis off,drawnow
        colormap hot
    end
end   