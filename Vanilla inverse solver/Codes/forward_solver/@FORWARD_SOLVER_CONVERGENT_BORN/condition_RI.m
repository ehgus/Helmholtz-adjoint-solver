function condition_RI(h)
    %add boundary to the RI
    h.create_boundary_RI();

    if size(h.RI,4)>1
        S = pagesvd(permute(h.potential(:),[4 5 1 2 3]));
        h.eps_imag =  max(abs(S(:)),[],'all').*1.01;
    else
        h.eps_imag = max(abs(h.potential),[],'all').*1.01;
    end
    step = abs(2*(2*pi*(h.RI_bg/h.wavelength))/h.eps_imag);
    h.pixel_step_size=round(step./(h.resolution));
    
    % h.potential = h.potential - 1i.*h.eps_imag
    if size(h.potential,4)==1
        h.potential = h.potential - 1i.*h.eps_imag;
    else
        for j1 = 1:3
            h.potential(:,:,:,j1,j1) = h.potential(:,:,:,j1,j1) - 1i.*h.eps_imag;
        end
    end
end