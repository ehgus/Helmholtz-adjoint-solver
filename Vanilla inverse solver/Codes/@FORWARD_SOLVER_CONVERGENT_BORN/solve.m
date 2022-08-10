function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
    if ~h.parameters.use_GPU
        input_field=single(input_field);
    else
        h.RI=single(gpuArray(h.RI));
        input_field=single(gpuArray(input_field));
    end
    
    assert(size(input_field,3)~=1 || ~h.parameters.vector_simulation, ...
    'Scalar simulation requires scalar field: size(input_field,3)==1');
    assert(size(input_field,3)~=2 || h.parameters.vector_simulation, ...
    'Vectorial simulation requires 2D vector field: size(input_field,3)==2');
    if h.parameters.verbose && size(input_field,3)==1
        warning('Scalar simulation is less precise');
    end
    
    input_field=fftshift(fft2(ifftshift(input_field)));
    %2D to 3D field
    [input_field] = h.transform_field_3D(input_field);
    %compute
    out_pol=1;
    if h.pole_num==3
        out_pol=2;
    end
    fields_trans=[];
    if h.parameters.return_transmission
        fields_trans=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
    end
    fields_ref=[];
    if h.parameters.return_reflection
        fields_ref=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
    end
    fields_3D=[];
    if h.parameters.return_3D
        fields_3D=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
    end
    
    for field_num=1:size(input_field,4)
        %figure; imagesc(abs(fftshift(ifft2(ifftshift(input_field(:,:,:,field_num))))));error('pause');
        Field=h.solve_raw(input_field(:,:,:,field_num));
        %crop and remove near field (3D to 2D field)
        if h.parameters.return_3D
            fields_3D(:,:,:,:,field_num)=gather(Field);
        end
        if h.parameters.return_transmission || h.parameters.return_reflection
            potential=RI2potential(h.RI(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:,:),h.parameters.wavelength,h.parameters.RI_bg);

            if size(h.RI,4)==1
                emitter_3D=Field.*potential.*h.utility_border.dV;
            else
                emitter_3D = 0;
                for j1 = 1:3
                    emitter_3D=emitter_3D+Field(:,:,:,j1).*potential(:,:,:,:,j1).*h.utility_border.dV;
                end
            end
            
            %emitter_3D=Field.*potential.*h.utility_border.dV;
            if ~h.cyclic_boundary_xy
            emitter_3D=h.padd_RI2conv(emitter_3D);
            end
            emitter_3D=fftshift(fft2(ifftshift(emitter_3D)));
        end
        if h.parameters.return_transmission
            if h.parameters.use_GPU
                h.kernel_trans=gpuArray(h.kernel_trans);
            end
            
            field_trans = h.crop_conv2field(fftshift(ifft2(ifftshift(sum(emitter_3D.*h.kernel_trans,3)))));
            field_trans=squeeze(field_trans);
            field_trans=fftshift(fft2(ifftshift(field_trans)));
            field_trans=field_trans+input_field(:,:,:,field_num);
            
            [field_trans] = h.transform_field_2D(field_trans);
            field_trans=fftshift(ifft2(ifftshift(field_trans)));
            fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
            h.kernel_trans=gather(h.kernel_trans);
        end
        if h.parameters.return_reflection
            if h.parameters.use_GPU
                h.kernel_ref=gpuArray(h.kernel_ref);
            end
            field_ref = h.crop_conv2field(fftshift(ifft2(ifftshift(sum(emitter_3D.*h.kernel_ref,3)))));
            field_ref=squeeze(field_ref);
            field_ref=fftshift(fft2(ifftshift(field_ref)));
            [field_ref] = h.transform_field_2D_reflection(field_ref);
            field_ref=fftshift(ifft2(ifftshift(field_ref)));
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
