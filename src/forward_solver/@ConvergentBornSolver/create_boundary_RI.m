function create_boundary_RI(h)
    % Pad boundary area for better simulation results
    old_RI_size=size(h.RI);
    
    pott=RI2potential(h.RI,h.wavelength,h.RI_bg);
    h.V=padarray(pott,h.boundary_thickness_pixel(1:3),'replicate');
    
    h.ROI = [...
        h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+old_RI_size(1)...
        h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+old_RI_size(2)...
        h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+old_RI_size(3)];
    
    h.attenuation_mask=cell(0);
    for dim = 1:3
        if h.boundary_thickness_pixel(dim)==0
            continue;
        end
        x=single(abs((1:size(h.V,dim))-(floor(size(h.V,dim)/2+1))+0.5)-0.5);
        x=circshift(x,-floor(size(h.V,dim)/2));
        x=x/(h.boundary_thickness_pixel(dim)-0.5);
        val0=x;val0(abs(x)>=1)=1;val0=abs(val0);
        val0=1-val0;
        val0(val0>h.boundary_sharpness)=h.boundary_sharpness;
        val0=val0./h.boundary_sharpness;
        h.attenuation_mask{end+1} = 1-reshape(val0,circshift([1 1 length(val0)],dim,2));
    end
end