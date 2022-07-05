function ROI = create_boundary_RI(h)
    
    
    % Pad boundary
    if (h.parameters.use_GPU)
        h.RI = gpuArray(h.RI);
    end
    old_RI_size=size(h.RI);
    
    h.RI=RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    %{
    h.RI=padarray(h.RI,...
        [h.boundary_thickness_pixel(1) h.boundary_thickness_pixel(2) h.boundary_thickness_pixel(3)],...
        h.parameters.RI_bg);
    %}
    h.RI=padarray(h.RI,...
        [h.boundary_thickness_pixel(1) h.boundary_thickness_pixel(2) h.boundary_thickness_pixel(3)],...
        0);
    h.RI=potential2RI(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    
    ROI = [...
        h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+old_RI_size(1)...
        h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+old_RI_size(2)...
        h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+old_RI_size(3)];
    %ROI = [1 ZP0(1) 1 ZP0(2) 1 ZP0(3)];
    
    V_temp = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
    
    h.attenuation_mask=1;
    for j1 = 1:3
        x=single(abs((1:size(V_temp,j1))-(floor(size(V_temp,j1)/2+1))+0.5)-0.5);x=circshift(x,-floor(size(V_temp,j1)/2));
        x=x/(h.boundary_thickness_pixel(j1)-0.5);
        %x=circshift(x,size(V_temp,j1)-round(h.boundary_thickness_pixel(j1)/2));
        val0=x;val0(abs(x)>=1)=1;val0=abs(val0);
        val0=1-val0;
        val0(val0>h.parameters.boundary_sharpness)=h.parameters.boundary_sharpness;
        val0=val0./h.parameters.boundary_sharpness;
        %figure; plot(val0); error('yo')
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
    %figure; plot(squeeze(h.attenuation_mask{end})),title('Axial boundary window attenuator strength');
    
    h.RI = potential2RI(V_temp,h.parameters.wavelength,h.parameters.RI_bg);
    if (h.parameters.use_GPU)
        h.RI=gather(h.RI);
        h.attenuation_mask=gather(h.attenuation_mask);
    end
end