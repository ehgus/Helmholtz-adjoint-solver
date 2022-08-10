function set_kernel(h)
    params_border=h.parameters;
    params_border.size=size(h.RI);
    warning('off','all');
    h.utility_border=DERIVE_OPTICAL_TOOL(params_border,h.parameters.use_GPU); % the utility for the space with border
    warning('on','all');
    
    if h.parameters.verbose && h.parameters.iterations_number>0
        warning('Best is to set iterations_number to -n for an automatic choice of this so that reflection to the order n-1 are taken in accound (transmission n=1, single reflection n=2, higher n=?)');
    end
    
    h.pole_num=1;
    if h.parameters.vector_simulation
        h.pole_num=3;
    end
    
    shifted_coordinate=h.utility_border.fourier_space.coor(1:3);
    if h.parameters.acyclic
        %shift all by kres/4
        if ~h.cyclic_boundary_xy
            %shift only in z h.parameters.acyclic
            shifted_coordinate{1}=shifted_coordinate{1}+h.utility_border.fourier_space.res{1}/4;
            shifted_coordinate{2}=shifted_coordinate{2}+h.utility_border.fourier_space.res{2}/4;
        end
        shifted_coordinate{3}=shifted_coordinate{3}+h.utility_border.fourier_space.res{3}/4;
    end
    for i=1:3
        shifted_coordinate{i}=ifftshift(shifted_coordinate{i});
    end
    
    if h.pole_num==3 % need to make true k/4 shift for rad !!!
        rads=...
            (shifted_coordinate{1}./h.utility_border.k0_nm).*reshape([1 0 0],1,1,1,[])+...
            (shifted_coordinate{2}./h.utility_border.k0_nm).*reshape([0 1 0],1,1,1,[])+...
            (shifted_coordinate{3}./h.utility_border.k0_nm).*reshape([0 0 1],1,1,1,[]);
    end
    
    h.green_absorbtion_correction=((2*pi*h.utility_border.k0_nm)^2)/((2*pi*h.utility_border.k0_nm)^2+1i.*h.eps_imag);
    
    % Maximum size of calculation of convergenent Born series
    h.Bornmax = h.parameters.iterations_number;
    if h.parameters.iterations_number==0
        warning('set iterations_number to either a positive or negative value');
    elseif h.parameters.iterations_number<0
        step = abs(2*(2*pi*h.utility_border.k0_nm)/h.eps_imag);
        Bornmax_opt = ceil(norm(size(h.RI,1:3).*h.parameters.resolution) / step / 2 + 1)*2; % -CHANGED
        h.Bornmax=Bornmax_opt*abs(h.parameters.iterations_number);
    end
    
    if h.parameters.verbose
        display(['number of step : ' num2str(h.Bornmax)])
        display(['step pixel size : ' num2str(h.pixel_step_size(3))])
    end
    
    eye_3=single(reshape(eye(3),1,1,1,3,3));

    % Helmholtz Green function in Fourier space 
    h.Greenp = 1 ./ (4*pi^2.*abs(...
        (shifted_coordinate{1}).^2 + ...
        (shifted_coordinate{2}).^2 + ...
        (shifted_coordinate{3}).^2 ...
        )-(2*pi*h.utility_border.k0_nm)^2-1i*h.eps_imag);
    
    % phase ramp
    if h.parameters.acyclic
        if all(h.boundary_thickness_pixel(1:2)==0)
            x=exp(-1i.*pi.*((1:size(h.V,3))-1)./size(h.V,3)/2);
            %x=circshift(x,-round(h.boundary_thickness_pixel/2));
            x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
            h.phase_ramp=reshape(x,1,1,[]);
        else
            for j1 = 1:3
                x=single(exp(-1i.*pi.*((1:size(h.V,j1))-1)./size(h.V,j1)/2));
                %x=circshift(x,-round(h.boundary_thickness_pixel(j1)/2));
                x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
                if j1 == 1
                    h.phase_ramp=reshape(x,[],1,1);
                elseif j1 == 2
                    h.phase_ramp=h.phase_ramp.*reshape(x,1,[],1);
                else
                    h.phase_ramp=h.phase_ramp.*reshape(x,1,1,[]);
                end
            end
        end
    else
        h.phase_ramp=1;
    end
    
    h.flip_Greenp = fft_flip(h.Greenp,[1 1 1],false);
    if h.pole_num==3 % dyadic Green's function
        [xsize, ysize, zsize] = size(h.Greenp);
        flip_rads = fft_flip(rads,[1 1 1],false);
        h.Greenp = h.Greenp.*(eye_3-h.green_absorbtion_correction*(rads).*reshape(rads,xsize,ysize,zsize,1,3));
        h.flip_Greenp = h.flip_Greenp.*(eye_3-h.green_absorbtion_correction*(flip_rads).*reshape(flip_rads,xsize,ysize,zsize,1,3));
    end
    
    h.Greenp = single(gather(h.Greenp));
    h.flip_Greenp = single(gather(h.flip_Greenp));
end