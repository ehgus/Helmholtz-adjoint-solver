function [Field, Hfield] = solve_forward(obj,incident_field)
    if (obj.use_GPU)
        obj.refocusing_util=gpuArray(obj.refocusing_util);
        incident_field=gpuArray(single(incident_field));
    end
    
    % generate incident field
    fourier_coord = cellfun(@(x) 2*pi*ifftshift(x),obj.utility.fourier_space.coor,'UniformOutput',false);
    fourier_coord_k3 = 2*pi*ifftshift(obj.utility.k3); 

    incident_field = ifft2(incident_field);
    incident_field = obj.padd_field2conv(incident_field);
    incident_field = fft2(incident_field);
    incident_field_h = zeros(size(incident_field),'like',incident_field);
    incident_field_h(:,:,1) = fourier_coord{2} .* incident_field(:,:,3) - fourier_coord_k3 .* incident_field(:,:,2);
    incident_field_h(:,:,2) = fourier_coord_k3 .* incident_field(:,:,1) - fourier_coord{1} .* incident_field(:,:,3);
    incident_field_h(:,:,3) = fourier_coord{1} .* incident_field(:,:,2) - fourier_coord{2} .* incident_field(:,:,1);
    incident_field_h = incident_field_h./(2*pi*obj.utility.nm/obj.wavelength);
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*obj.refocusing_util;
    incident_field = ifft2(incident_field);
    incident_field = obj.crop_conv2RI(incident_field);
    incident_field_h = reshape(incident_field_h, [size(incident_field_h,1),size(incident_field_h,2),1,size(incident_field_h,3)]).*obj.refocusing_util;
    incident_field_h = ifft2(incident_field_h);
    incident_field_h = obj.crop_conv2RI(incident_field_h);
    impedance = 377/obj.utility.nm;
    incident_field_h = incident_field_h/impedance;
    % place incident field at the edge of the simulation region
    % The position is fixed for now, but it can move any place in the future
    incident_field(:,:,1:obj.ROI(5)-1,:) =flip(incident_field(:,:,obj.ROI(5):2*obj.ROI(5)-2,:),3);

    obj.refocusing_util=gather(obj.refocusing_util);
    
    % Evaluate output field
    [Field, Hfield] = eval_scattered_field(obj,incident_field);
    Field = Field + gather(incident_field(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:));
    Hfield = Hfield + gather(incident_field_h(obj.ROI(1):obj.ROI(2),obj.ROI(3):obj.ROI(4),obj.ROI(5):obj.ROI(6),:));
end