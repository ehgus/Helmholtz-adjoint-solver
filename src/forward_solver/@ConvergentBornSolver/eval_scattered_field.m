function [Field, Hfield] =eval_scattered_field(obj,source)
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

    psi = complex(zeros(size_field,array_option{:}));
    PSI = complex(zeros(size_field,array_option{:}));
    Field = complex(zeros(size_field,array_option{:}));
    Field_n = complex(zeros(size_field,array_option{:}));

    isacyclic = ~all(obj.periodic_boudnary);
    for Born_order = 1:obj.Bornmax
        if isacyclic
            %flip the relevant quantities
            [Green_fn, flip_Green_fn] = deal(flip_Green_fn, Green_fn);
        end
        
        if Born_order <= 2 % source
            Field_n(:)=0;
            psi(:) = (1i/obj.eps_imag/4)*source;
            for idx = 1:length(obj.field_attenuation_mask)
                % Attenuation
                psi=psi.*obj.field_attenuation_mask{idx};
            end
        else % gamma * E
            if is_isotropic
                psi = obj.V .* Field_n;
            else
                psi(:) = 0;
                for axis = 1:3
                    psi = psi + obj.V(:,:,:,:,axis) .* Field_n(:,:,:,axis);
                end
            end
            psi = (1i/obj.eps_imag)*psi;
            Field_n = Field_n - psi;
        end
        % Convolution with G
        PSI = conv(Green_fn, psi, PSI);
        % element-wise multiplication of V
        if is_isotropic
            Field_n = Field_n + obj.V .* PSI;
        else
            for axis = 1:3
                Field_n = Field_n + obj.V(:,:,:,:,axis) .* PSI(:,:,:,axis);
            end
        end
        % Attenuation
        for idx = 1:length(obj.field_attenuation_mask)
            Field_n=Field_n.*obj.field_attenuation_mask{idx};
        end
        % add the fields to the total field
        if Born_order == 3
            Field_n = Field_n + Field;
        end
        Field = Field + Field_n;
        if Born_order == 2
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