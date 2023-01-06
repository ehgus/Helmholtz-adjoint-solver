function Field=eval_scattered_field(h,incident_field)
    % Convergence Born series
    % E1 = r*G*S/4
    % E2 = r*G_flip*S/4
    % E3 = M * (E1 + E2) + (E1 + E2)
    % E_(j+1) = M E_j
    pole_num = 1;
    if h.vector_simulation
        pole_num = 3;
    end
    size_field=[size(h.V,1),size(h.V,2),size(h.V,3), pole_num];
    if (h.use_GPU)
        h.V=gpuArray(h.V);
        h.phase_ramp=gpuArray(h.phase_ramp);
        h.attenuation_mask=gpuArray(h.attenuation_mask);
        h.Greenp = gpuArray(h.Greenp);
        h.flip_Greenp = gpuArray(h.flip_Greenp);
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
    if h.acyclic
        flip_Greenp = h.flip_Greenp;
    end
    phase_ramp = h.phase_ramp;
    conj_phase_ramp=conj(h.phase_ramp);
    
    for jj = 1:h.Bornmax
        if h.acyclic
            %flip the relevant quantities
            [Greenp, flip_Greenp] = deal(flip_Greenp, Greenp);
            [phase_ramp, conj_phase_ramp] = deal(conj_phase_ramp, phase_ramp);
        end
        
        %init other quantities
        PSI(:) = 0;
        psi(:) = 0;
        
        if jj <= 2 % s
            Field_n(:)=0;
            psi(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) = (1i/h.eps_imag/4).*source;
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
        for j2 = 1:pole_num
            coeff_field=fftn(psi(:,:,:,j2).*phase_ramp);
            if h.vector_simulation %dyadic absorptive green function convolution
                PSI = PSI + Greenp(:,:,:,:,j2).* coeff_field;
            else %scalar absorptive green function convolution
                PSI = Greenp.*coeff_field;
            end
        end
        for j1 = 1:pole_num
            PSI(:,:,:,j1)=ifftn(PSI(:,:,:,j1)).*conj_phase_ramp;
        end
        if jj > 2
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
        %Field_n=Field_n.*h.attenuation_mask;
        
        % add the fields to the total field
        if jj==3
            Field_n = Field_n + Field;
        end
        Field = Field + Field_n;
        if jj==2
            Field_n = Field;
        end
    end
    
    Field = Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:);
end