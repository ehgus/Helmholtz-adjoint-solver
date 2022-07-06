function Field=solve_adjoint(h,source, intensity_mask)
    size_field=[size(h.V,1),size(h.V,2),size(h.V,3),h.pole_num];
    if (h.parameters.use_GPU)
        h.V=gpuArray(h.V);
        h.refocusing_util=gpuArray(h.refocusing_util);
        h.phase_ramp=gpuArray(h.phase_ramp);
        source=gpuArray(single(source));
        h.attenuation_mask=gpuArray(h.attenuation_mask);
        h.rads = gpuArray(h.rads);
        h.eye_3 = gpuArray(h.eye_3);
        h.Greenp = gpuArray(h.Greenp);
        psi = zeros(size_field,'single','gpuArray');
        PSI = zeros(size_field,'single','gpuArray');
        Field = zeros(size_field,'single','gpuArray');
        Field_n = zeros(size_field,'single','gpuArray');
    else
        psi = zeros(size_field,'single');
        PSI = zeros(size_field,'single');
        Field = zeros(size_field,'single');
        Field_n = zeros(size_field,'single');
    end
    
    h.refocusing_util=gather(h.refocusing_util);
    source = source.*intensity_mask;

    incident_field = source;
    if size(h.RI,4)==1
        source = (h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6))+1i*h.eps_imag).*source;
    else
        source00 = source;
        source(:) = 0;
        for j1 = 1:3
            source = source + (h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)):(h.ROI(6)),:,j1)) .* source00(:,:,:,j1);
        end
        source = source+1i*h.eps_imag*source00;
        clear source00
    end
    
    for jj = 1:h.Bornmax
        %flip the relevant quantities
        if h.parameters.acyclic
            h.Greenp=fft_flip(h.Greenp,[1 1 1],false);
            if h.pole_num==3
                h.rads=fft_flip(h.rads,[1 1 1],false);
            end
            h.phase_ramp=conj(h.phase_ramp);
        end
        
        %init other quantities
        PSI(:) = 0;
        psi(:) = 0;
        
        if any(jj == [1,2]) % s
            Field_n(:)=0;
            psi(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) = (1i./ h.eps_imag).*source/2;
        else % gamma * E
            if size(h.V,4) == 1
                psi = (1i./h.eps_imag) .* Field_n .* h.V;
            else
                for j1 = 1:3
                    psi = psi + (1i./h.eps_imag) .* Field_n(:,:,:,j1) .* h.V(:,:,:,:,j1);
                end
            end
        end
        
        % G x
        for j2 = 1:h.pole_num
            coeff_field=fftn(psi(:,:,:,j2).*h.phase_ramp);
            if h.pole_num==3 %dyadic absorptive green function convolution
                PSI = PSI + ((h.Greenp.*(h.eye_3(:,:,:,:,j2)-h.green_absorbtion_correction*(h.rads).*(h.rads(:,:,:,j2)))).* coeff_field);
            else %dscalar absorptive green function convolution
                PSI = (h.Greenp.*coeff_field);
            end
        end
        for j1 = 1:h.pole_num
            PSI(:,:,:,j1)=ifftn(PSI(:,:,:,j1)).*conj(h.phase_ramp);
        end
        if ~any(jj == [1,2])
            Field_n = Field_n - psi;
        end
        if size(h.V,4) == 1
            Field_n = Field_n + (h.V) .* PSI;
        else
            for j1 = 1:3
                Field_n = Field_n + (h.V(:,:,:,:,j1)) .* PSI(:,:,:,j1);
            end
        end
        % Attenuation
        Field_n=Field_n.*h.attenuation_mask;
        
        % add the fields to the total field
        if jj==2
            clear source;
            temp=Field;
        end
        Field = Field + Field_n;
        if jj==3
            Field_n=Field_n+temp;
            clear temp;
        end
    end
    
    Field = ...
        Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) + incident_field;

    if h.parameters.verbose
        set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]), colorbar, axis off,drawnow
        colormap hot
    end
end   