function Field=eval_scattered_field(h,incident_field)
    size_field=[size(h.V,1),size(h.V,2),size(h.V,3),h.pole_num];
    if (h.parameters.use_GPU)
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
    
    if size(h.RI,4)==1 % scalar RI = (x,y,z, tensor_dim1, tensor_dim2)
        source = (h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6))+1i*h.eps_imag).*incident_field;
    else % tensor
        source = zeros('like',incident_field);
        for j1 = 1:3
            source = source + h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)):(h.ROI(6)),:,j1) .* incident_field(:,:,:,j1);
        end
        source = soruce + 1i*h.eps_imag .* incident_field;
    end
    
    Greenp = h.Greenp;
    if h.parameters.acyclic
        flip_Greenp = h.flip_Greenp;
    end
    phase_ramp = h.phase_ramp;
    conj_phase_ramp=conj(h.phase_ramp);
    
    for jj = 1:h.Bornmax
        %flip the relevant quantities
        if h.parameters.acyclic
            [Greenp, flip_Greenp] = deal(flip_Greenp, Greenp);
            [phase_ramp, conj_phase_ramp] = deal(conj_phase_ramp, phase_ramp);
        end
        
        %init other quantities
        PSI(:) = 0;
        psi(:) = 0;
        
        if any(jj == [1,2]) % s
            Field_n(:)=0;
            psi(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) = (1i./h.eps_imag).*source/2;
        else % gamma * E
            if size(h.V,4) == 1
                psi = h.V .* Field_n;
            else
                for j1 = 1:3
                    psi = psi + h.V(:,:,:,:,j1) .* Field_n(:,:,:,j1);
                end
            end
            psi = (1i./h.eps_imag).*psi;
        end
        
        % G x
        for j2 = 1:h.pole_num
            coeff_field=fftn(psi(:,:,:,j2).*phase_ramp);
            if h.pole_num==3 %dyadic absorptive green function convolution
                PSI = PSI + Greenp(:,:,:,:,j2).* coeff_field;
            else %scalar absorptive green function convolution
                PSI = Greenp.*coeff_field;
            end
        end
        for j1 = 1:h.pole_num
            PSI(:,:,:,j1)=ifftn(PSI(:,:,:,j1)).*conj_phase_ramp;
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
    
    Field = Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:);
end