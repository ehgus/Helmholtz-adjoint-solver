function condition_RI(h)
    %add boundary to the RI
    h.create_boundary_RI();

    if size(h.RI,4)>1
        S = pagesvd(permute(h.V(:),[4 5 1 2 3]));
        h.eps_imag =  max(abs(S(:)),[],'all').*1.01;
    else
        h.eps_imag = max(abs(h.V),[],'all').*1.01;
    end
    % h.V = h.V - 1i.*h.eps_imag
    if size(h.V,4)==1
        h.V = h.V - 1i.*h.eps_imag;
    else
        for j1 = 1:3
            h.V(:,:,:,j1,j1) = h.V(:,:,:,j1,j1) - 1i.*h.eps_imag;
        end
    end
end