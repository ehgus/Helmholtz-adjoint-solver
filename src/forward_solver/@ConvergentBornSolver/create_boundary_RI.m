function create_boundary_RI(h)
    % Pad boundary area for better simulation results
    pott=RI2potential(h.RI,h.wavelength,h.RI_bg);
    h.V=padarray(pott,h.boundary_thickness_pixel(1:3),'replicate');
end