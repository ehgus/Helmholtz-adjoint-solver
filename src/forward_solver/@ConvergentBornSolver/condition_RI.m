function condition_RI(h)
    %add boundary to the RI
    h.create_boundary_RI();

    if size(h.V,4)>1
        S = pagesvd(permute(h.V(:),[4 5 1 2 3]));
        h.eps_imag =  max(abs(S(:)),[],'all').*1.01;
    else
        h.eps_imag = max(abs(h.V),[],'all').*1.01;
    end
    h.eps_imag = max(2^-20,h.eps_imag);
    % h.V = h.V - 1i.*h.eps_imag
    if size(h.V,4)==1
        h.V = h.V - 1i.*h.eps_imag;
    else
        for j1 = 1:3
            h.V(:,:,:,j1,j1) = h.V(:,:,:,j1,j1) - 1i.*h.eps_imag;
        end
    end

    % apply edge filter
    % See appendix B of "Ultra-thin boundary layer for high-accuracy simulation of light propagation"
    for dim = 1:3
        max_L = h.boundary_thickness_pixel(dim);
        L = min(max_L, h.potential_attenuation_pixel(dim));
        if L == 0
            continue
        end
        window = (tanh(linspace(-3,3,L))/tanh(3)-tanh(-3))/2;
        window = window*h.potential_attenuation_sharpness + (1-h.potential_attenuation_sharpness);

        filter =[window ones(1, h.ROI(2*(dim-1)+2) - h.ROI(2*(dim-1)+1) + 1 + 2*(max_L-L)) flip(window)];
        filter = reshape(filter, circshift([1, 1, length(filter)], [0 dim]));
        h.V = h.V .* filter;
    end
end