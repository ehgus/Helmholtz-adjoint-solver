function rst = curl_field(obj, field)
    % curl operator
    rst = zeros(size(field),'like',field);
    warning('off','all');
    obj.utility = derive_utility(obj, size(rst,1:3)); % the utility for the space with border
    warning('on','all');
    % 2pi*i/L * grid number = fourier_coord
    fourier_coord = cellfun(@(x) 2i*pi*ifftshift(x),obj.utility.fourier_space.coor,'UniformOutput',false);
    for axis = 1:3
        ordered_axis = circshift([1,2,3],-axis);
        cross_fft = fftn(field(:,:,:,ordered_axis(3)));
        rst(:,:,:,ordered_axis(1)) = rst(:,:,:,ordered_axis(1)) +  fourier_coord{ordered_axis(2)} .* cross_fft;
        rst(:,:,:,ordered_axis(2)) = rst(:,:,:,ordered_axis(2)) -  fourier_coord{ordered_axis(1)} .* cross_fft;
    end
    for axis = 1:3
        rst(:,:,:,axis) = ifftn(rst(:,:,:,axis));
    end
end