function [fields_3D,Hfields]=solve(h,input_field)
    if ~h.use_GPU
        input_field=single(input_field);
    else
        input_field=single(gpuArray(input_field));
    end

    if size(input_field,3) == 2
        error('The 3rd dimension of input_field should indicate polarization');
    end
    if h.verbose && size(input_field,3)==1
        warning('Scalar simulation is less precise');
    end

    input_field=fft2(input_field);
    %2D to 3D field
    input_field = ifftshift2(h.transform_field_3D(fftshift2(input_field)));
    %compute
    fields_3D=[];
    if h.return_3D
        fields_3D=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
    end
    Hfields=[];
    if h.return_3D
        Hfields=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
    end

    for field_num=1:size(input_field,4)
        [Field, Hfield]=h.solve_forward(input_field(:,:,:,field_num));
        %crop and remove near field (3D to 2D field)
        if h.return_3D
            fields_3D(:,:,:,:,field_num)=gather(Field);
            Hfields(:,:,:,:,field_num)=gather(Hfield);
        end
        if h.return_transmission || h.return_reflection
            potential = h.V(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:,:);
            if size(h.V,4)==1
                potential = potential + 1i.*h.eps_imag;
                emitter_3D=Field.*potential;
            else
                for j1 = 1:3
                    potential(:,:,:,j1,j1) = potential(:,:,:,j1,j1) + 1i.*h.eps_imag;
                end
                emitter_3D = 0;
                for j1 = 1:3
                    emitter_3D=emitter_3D+Field(:,:,:,j1).*potential(:,:,:,:,j1);
                end
            end
            emitter_3D = emitter_3D*h.utility.dV;
            if ~h.cyclic_boundary_xy
            emitter_3D=h.padd_RI2conv(emitter_3D);
            end
            if h.use_GPU
                emitter_3D=gpuArray(emitter_3D);
            end
            emitter_3D=fft2(ifftshift2(emitter_3D));
        end
    end
    %gather to release gpu memory
    h.V=gather(h.V);
    h.field_attenuation_mask=gather(h.field_attenuation_mask);
    h.phase_ramp=gather(h.phase_ramp);
end
