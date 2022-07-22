function ROI = create_boundary_RI(h)
    
    % Pad boundary
    old_RI_size=size(h.RI);
    
    pott=RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    pott=padarray(pott,h.boundary_thickness_pixel(1:3),0);
    h.RI=potential2RI(pott,h.parameters.wavelength,h.parameters.RI_bg);
    
    ROI = [...
        h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+old_RI_size(1)...
        h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+old_RI_size(2)...
        h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+old_RI_size(3)];
    
    pott = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    
    h.attenuation_mask=ones(size(pott),'single');
    for j1 = 1:3
        x=single(abs((1:size(pott,j1))-(floor(size(pott,j1)/2+1))+0.5)-0.5);x=circshift(x,-floor(size(pott,j1)/2));
        x=x/(h.boundary_thickness_pixel(j1)-0.5);
        %x=circshift(x,size(V_temp,j1)-round(h.boundary_thickness_pixel(j1)/2));
        val0=x;val0(abs(x)>=1)=1;val0=abs(val0);
        val0=1-val0;
        val0(val0>h.parameters.boundary_sharpness)=h.parameters.boundary_sharpness;
        val0=val0./h.parameters.boundary_sharpness;
        if h.boundary_thickness_pixel(j1)==0
            val0(:)=0;
        end
        if j1 == 1
            h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,[],1,1).*1);
        elseif j1 == 2
            h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,1,[],1).*1);
        else
            h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,1,1,[]).*1);
        end
    end
    
    h.RI = potential2RI(pott,h.parameters.wavelength,h.parameters.RI_bg);
end