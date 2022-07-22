function Field=solve_adjoint(h,propagating_field, intensity_mask)
    if (h.parameters.use_GPU)
        h.V=gpuArray(h.V);
        h.phase_ramp=gpuArray(h.phase_ramp);
        propagating_field=gpuArray(single(propagating_field));
        h.attenuation_mask=gpuArray(h.attenuation_mask);
        h.Greenp = gpuArray(h.Greenp);
        h.flip_Greenp = gpuArray(h.flip_Greenp);
    end

    incident_field = propagating_field.*intensity_mask;
    
    % Evaluate output field
    Field = eval_scattered_field(h, incident_field);
    Field = Field + incident_field;

    if h.parameters.verbose
        set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]), colorbar, axis off,drawnow
        colormap hot
    end
end   