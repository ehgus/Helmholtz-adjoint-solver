function condition_RI(obj)
    %add boundary to the RI
    obj.create_boundary_RI();

    if size(obj.V,4)>1
        S = pagesvd(permute(obj.V(:),[4 5 1 2 3]));
        obj.eps_imag =  max(abs(S(:)),[],'all').*1.01;
    else
        obj.eps_imag = max(abs(obj.V),[],'all').*1.01;
    end
    obj.eps_imag = max(2^-20,obj.eps_imag);
    % obj.V = obj.V - 1i.*obj.eps_imag
    if size(obj.V,4)==1
        obj.V = obj.V - 1i.*obj.eps_imag;
    else
        for axis = 1:3
            obj.V(:,:,:,axis,axis) = obj.V(:,:,:,axis,axis) - 1i.*obj.eps_imag;
        end
    end

    % apply edge filter
    % See appendix B of "Ultra-thin boundary layer for high-accuracy simulation of light propagation"
    for dim = 1:3
        max_L = obj.boundary_thickness_pixel(dim);
        L = min(max_L, obj.potential_attenuation_pixel(dim));
        if L == 0
            continue
        end
        window = (tanh(linspace(-3,3,L))/tanh(3)-tanh(-3))/2;
        window = window*obj.potential_attenuation_sharpness + (1-obj.potential_attenuation_sharpness);

        filter =[window ones(1, obj.ROI(2*(dim-1)+2) - obj.ROI(2*(dim-1)+1) + 1 + 2*(max_L-L)) flip(window)];
        filter = reshape(filter, circshift([1, 1, length(filter)], [0 dim]));
        obj.V = obj.V .* filter;
    end
end