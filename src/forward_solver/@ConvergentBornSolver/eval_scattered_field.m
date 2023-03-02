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
    is_isotropic = size(h.V, 4) == 1;  % scalar RI = (x,y,z, tensor_dim1, tensor_dim2)
    size_field=[size(h.V,1),size(h.V,2),size(h.V,3), pole_num];

    if h.use_GPU
        array_func = @gpuArray;
        array_option = {'single','gpuArray'};
    else
        array_func = @(x) x;
        array_option = {'single'};
    end
    h.V = array_func(h.V);
    for idx = 1:length(h.attenuation_mask)
        h.attenuation_mask{idx}=array_func(h.attenuation_mask{idx});
    end
    Greenp = array_func(h.Greenp);
    flip_Greenp = array_func(h.flip_Greenp);
    phase_ramp = cell(1,length(h.phase_ramp));
    conj_phase_ramp = cell(1,length(h.phase_ramp));
    for idx = 1:length(phase_ramp)
        phase_ramp{idx} = array_func(h.phase_ramp{idx});
        conj_phase_ramp{idx} = conj(phase_ramp{idx});
    end
    psi = zeros(size_field,array_option{:});
    PSI = zeros(size_field,array_option{:});
    Field = zeros(size_field,array_option{:});
    Field_n = zeros(size_field,array_option{:});
    
    if is_isotropic
        source = h.V .* incident_field;
    else % tensor
        source = zeros('like',incident_field);
        for j1 = 1:3
            source = source + h.V(:,:,:,:,j1) .* incident_field(:,:,:,j1);
        end
    end
    source = source + 1i*h.eps_imag * incident_field;
    
    for jj = 1:h.Bornmax
        if h.acyclic
            %flip the relevant quantities
            [Greenp, flip_Greenp] = deal(flip_Greenp, Greenp);
            [phase_ramp, conj_phase_ramp] = deal(conj_phase_ramp, phase_ramp);
        end
        
        %init other quantities
        PSI(:) = 0;
        psi(:) = 0;
        
        if jj <= 2 % source
            Field_n(:)=0;
            psi(:) = (1i/h.eps_imag/4)*source;
        else % gamma * E
            if is_isotropic
                psi = h.V .* Field_n;
            else
                for j1 = 1:3
                    psi = psi + h.V(:,:,:,:,j1) .* Field_n(:,:,:,j1);
                end
            end
            psi = (1i/h.eps_imag)*psi;
            Field_n = Field_n - psi;
        end
        % G x
        for idx = 1:length(phase_ramp)
            psi = psi.*phase_ramp{idx};
        end
        for j2 = 1:pole_num
            psi(:,:,:,j2)=fftn(psi(:,:,:,j2));
            if h.vector_simulation %dyadic absorptive green function convolution
                PSI = PSI + Greenp(:,:,:,:,j2).* psi(:,:,:,j2);
            else %scalar absorptive green function convolution
                PSI = Greenp.*psi(:,:,:,j2);
            end
        end
        for j1 = 1:pole_num
            PSI(:,:,:,j1)=ifftn(PSI(:,:,:,j1));
        end
        for idx = 1:length(phase_ramp)
            PSI = PSI.*conj_phase_ramp{idx};
        end
        
        if is_isotropic
            Field_n = Field_n + h.V .* PSI;
        else
            for j1 = 1:3
                Field_n = Field_n + h.V(:,:,:,:,j1) .* PSI(:,:,:,j1);
            end
        end
        % Attenuation
        for idx = 1:length(h.attenuation_mask)
            Field_n=Field_n.*h.attenuation_mask{idx};
        end
        % add the fields to the total field
        if jj==3
            Field_n = Field_n + Field;
        end
        Field = Field + Field_n;
        if jj==2
            Field_n = Field;
        end
    end
    
    Field = gather(Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:));
    h.V=gather(h.V);
    h.attenuation_mask=gather(h.attenuation_mask);
end