function [fields_trans,fields_ref,fields_3D,Hfields]=solve(h,input_field)
    if ~h.use_GPU
        input_field=single(input_field);
    else
        input_field=single(gpuArray(input_field));
    end

    assert(size(input_field,3)~=1 || ~h.vector_simulation, ...
    'Scalar simulation requires scalar field: size(input_field,3)==1');
    assert(size(input_field,3)~=2 || h.vector_simulation, ...
    'Vectorial simulation requires 2D vector field: size(input_field,3)==2');
    if h.verbose && size(input_field,3)==1
        warning('Scalar simulation is less precise');
    end

    input_field=fft2(input_field);
    %2D to 3D field
    input_field = ifftshift2(h.transform_field_3D(fftshift2(input_field)));
    %compute
    out_pol=1;
    if h.vector_simulation
        out_pol=2;
    end
    fields_trans=[];
    if h.return_transmission
        fields_trans=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
    end
    fields_ref=[];
    if h.return_reflection
        fields_ref=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
    end
    fields_3D=[];
    if h.return_3D
        fields_3D=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
    end
    Hfields=[];
    if h.return_3D && h.vector_simulation
        Hfields=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
    end

    for field_num=1:size(input_field,4)
        [Field, Hfield]=h.solve_forward(input_field(:,:,:,field_num));
        %crop and remove near field (3D to 2D field)
        if h.return_3D
            fields_3D(:,:,:,:,field_num)=gather(Field);
            if h.vector_simulation
                Hfields(:,:,:,:,field_num)=gather(Hfield);
            end
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
        if h.return_transmission
            if h.use_GPU
                h.kernel_trans=gpuArray(h.kernel_trans);
            end

            field_trans = h.crop_conv2field(fftshift2(ifft2(sum(emitter_3D.*fftshift(h.kernel_trans,3),3))));
            field_trans=squeeze(field_trans);
            field_trans=fft2(field_trans);
            field_trans=field_trans + input_field(:,:,:,field_num);

            field_trans = ifftshift2(h.transform_field_2D(fftshift2(field_trans)));
            field_trans=ifft2(field_trans);
            fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
            h.kernel_trans=gather(h.kernel_trans);
        end
        if h.return_reflection
            if h.use_GPU
                h.kernel_ref=gpuArray(h.kernel_ref);
            end
            field_ref = h.crop_conv2field(fftshift2(ifft2(sum(emitter_3D.*fftshift(h.kernel_ref,3),3))));
            field_ref=squeeze(field_ref);
            field_ref=fft2(field_ref);
            field_ref = ifftshift2(h.transform_field_2D_reflection(fftshift2(field_ref)));
            field_ref=ifft2(field_ref);
            fields_ref(:,:,:,field_num)=gather(squeeze(field_ref));

            h.kernel_ref=gather(h.kernel_ref);
        end
    end
    %gather to release gpu memory
    h.V=gather(h.V);
    h.attenuation_mask=gather(h.attenuation_mask);
    h.phase_ramp=gather(h.phase_ramp);
    h.Greenp = gather(h.Greenp);
    h.flip_Greenp = gather(h.flip_Greenp);
end
