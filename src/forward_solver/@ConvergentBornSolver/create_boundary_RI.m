function create_boundary_RI(obj)
    % Pad boundary area for better simulation results
    pott=RI2potential(obj.RI,obj.wavelength,obj.RI_bg);
    obj.V=padarray(pott,obj.boundary_thickness_pixel(1:3),'replicate');
end