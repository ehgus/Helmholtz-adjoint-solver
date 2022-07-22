function condition_RI(h)
    if size(h.RI,4)>1
        pott=RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
        S = pagesvd(permute(pott(:),[4 5 1 2 3]));
        h.eps_imag =  max(abs(S(:)),[],'all').*1.01;
    else
        h.eps_imag = max(abs(RI2potential(h.RI(:),h.parameters.wavelength,h.parameters.RI_bg)),[],'all').*1.01;
    end
    step = abs(2*(2*pi*(h.parameters.RI_bg/h.parameters.wavelength))/h.eps_imag);
    h.pixel_step_size=round(step./(h.parameters.resolution));
    %add boundary to the RI
    h.ROI = h.create_boundary_RI(); %note: create_boundary_RI implicitely pad the h.RI along z axis
    
    %update the size in the parameters
    h.V = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    % h.V = h.V - 1i.*h.eps_imag
    if size(h.V,4)==1
        h.V = h.V - 1i.*h.eps_imag;
    else
        for j1 = 1:3
            h.V(:,:,:,j1,j1) = h.V(:,:,:,j1,j1) - 1i.*h.eps_imag;
        end
    end
end