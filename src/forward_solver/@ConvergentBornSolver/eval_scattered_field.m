function [Field, Hfield] =eval_scattered_field(obj,incident_field)
    % Convergence Born series
    % E1 = r*G*S/4
    % E2 = r*G_flip*S/4
    % E3 = M * (E1 + E2) + (E1 + E2)
    % E_(j+1) = M E_j
    is_isotropic = size(obj.V, 4) == 1;  % scalar RI = (x,y,z, tensor_dim1, tensor_dim2)
    size_field=[size(obj.V,1),size(obj.V,2),size(obj.V,3), 3];

    if obj.use_GPU
        array_func = @gpuArray;
        array_option = {'single','gpuArray'};
    else
        array_func = @(x) x;
        array_option = {'single'};
    end
    obj.V = array_func(obj.V);
    for idx = 1:length(obj.field_attenuation_mask)
        obj.field_attenuation_mask{idx}=array_func(obj.field_attenuation_mask{idx});
    end
    Green_fn = obj.Green_fn;
    flip_Green_fn = obj.flip_Green_fn;
    phase_ramp = cell(1,length(obj.phase_ramp));
    conj_phase_ramp = cell(1,length(obj.phase_ramp));
    for idx = 1:length(phase_ramp)
        phase_ramp{idx} = array_func(obj.phase_ramp{idx});
        conj_phase_ramp{idx} = conj(phase_ramp{idx});
    end
    psi = zeros(size_field,array_option{:});
    PSI = zeros(size_field,array_option{:});
    Field = zeros(size_field,array_option{:});
    Field_n = zeros(size_field,array_option{:});
    
    if is_isotropic
        source = obj.V .* incident_field;
    else % tensor
        source = zeros('like',incident_field);
        for j1 = 1:3
            source = source + obj.V(:,:,:,:,j1) .* incident_field(:,:,:,j1);
        end
    end
    source = source + 1i*obj.eps_imag * incident_field;
    % Attenuation
    for idx = 1:length(obj.field_attenuation_mask)
        source=source.*obj.field_attenuation_mask{idx};
    end
    for jj = 1:obj.Bornmax
        if obj.acyclic
            %flip the relevant quantities
            [Green_fn, flip_Green_fn] = deal(flip_Green_fn, Green_fn);
            [phase_ramp, conj_phase_ramp] = deal(conj_phase_ramp, phase_ramp);
        end
        
        %init other quantities
        PSI(:) = 0;
        psi(:) = 0;
        
        if jj <= 2 % source
            Field_n(:)=0;
            psi(:) = (1i/obj.eps_imag/4)*source;
        else % gamma * E
            if is_isotropic
                psi = obj.V .* Field_n;
            else
                for j1 = 1:3
                    psi = psi + obj.V(:,:,:,:,j1) .* Field_n(:,:,:,j1);
                end
            end
            psi = (1i/obj.eps_imag)*psi;
            Field_n = Field_n - psi;
        end
        % multiply G
        for idx = 1:length(phase_ramp)
            psi = psi.*phase_ramp{idx};
        end
        PSI = Green_fn(PSI, psi);
        for idx = 1:length(phase_ramp)
            PSI = PSI.*conj_phase_ramp{idx};
        end
        % multiply V
        if is_isotropic
            Field_n = Field_n + obj.V .* PSI;
        else
            for j1 = 1:3
                Field_n = Field_n + obj.V(:,:,:,:,j1) .* PSI(:,:,:,j1);
            end
        end
        % Attenuation
        for idx = 1:length(obj.field_attenuation_mask)
            Field_n=Field_n.*obj.field_attenuation_mask{idx};
        end
        % add the fields to the total field
        if jj==3
            Field_n = Field_n + Field;
        end
        Field = Field + Field_n;
        if jj==2
            Field_n = Field;
        end
        if obj.verbose
            figure(101)
            imagesc(squeeze(gather(abs(Field(1,:,:,1)))))
            drawnow
        end
    end
    % H = -i/k_0 * (n_0/impedance_0) * curl(E)
    Hfield = obj.curl_field(Field);
    Hfield = -1i * obj.wavelength/(2*pi*377) .* gather(Hfield(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:,:));

    Field = gather(Field(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:,:));
    obj.V=gather(obj.V);
    obj.field_attenuation_mask=gather(obj.field_attenuation_mask);
end