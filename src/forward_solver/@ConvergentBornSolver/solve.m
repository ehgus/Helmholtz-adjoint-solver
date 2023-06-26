function [fields_3D,Hfields]=solve(obj, input_field)
    if ~obj.use_GPU
        input_field=single(input_field);
    else
        input_field=single(gpuArray(input_field));
    end

    if size(input_field,3) == 2
        error('The 3rd dimension of input_field should indicate polarization');
    end
    if obj.verbose && size(input_field,3)==1
        warning('Scalar simulation is less precise');
    end

    input_field=fft2(input_field);
    %2D to 3D field
    input_field = ifftshift2(obj.transform_field_3D(fftshift2(input_field)));
    %compute
    fields_3D=[];
    if obj.return_3D
        fields_3D=ones(1+obj.ROI(2)-obj.ROI(1),1+obj.ROI(4)-obj.ROI(3),1+obj.ROI(6)-obj.ROI(5),size(input_field,3),size(input_field,4),'single');
    end
    Hfields=[];
    if obj.return_3D
        Hfields=ones(1+obj.ROI(2)-obj.ROI(1),1+obj.ROI(4)-obj.ROI(3),1+obj.ROI(6)-obj.ROI(5),size(input_field,3),size(input_field,4),'single');
    end

    for field_num=1:size(input_field,4)
        [Field, Hfield]=obj.solve_forward(input_field(:,:,:,field_num));
        %crop and remove near field (3D to 2D field)
        if obj.return_3D
            fields_3D(:,:,:,:,field_num)=gather(Field);
            Hfields(:,:,:,:,field_num)=gather(Hfield);
        end
    end
    %gather to release gpu memory
    obj.V=gather(obj.V);
    obj.field_attenuation_mask=gather(obj.field_attenuation_mask);
    obj.phase_ramp=gather(obj.phase_ramp);
end
