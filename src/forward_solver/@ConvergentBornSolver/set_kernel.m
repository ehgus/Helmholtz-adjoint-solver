function set_kernel(h)
    Nsize = size(h.V);
    warning('off','all');
    h.utility = derive_utility(h, Nsize); % the utility for the space with border
    warning('on','all');
    
    % Maximum size of calculation of convergenent Born series
    if h.iterations_number==0
        error('set iterations_number to either a positive or negative value');
    elseif h.iterations_number<0
        step = abs(2*(2*pi*h.utility.k0_nm)/h.eps_imag);
        Bornmax_opt = ceil(norm(size(h.V,1:3).*h.resolution) / step / 2 + 1)*2;
        h.Bornmax=Bornmax_opt*abs(h.iterations_number);
    else
        warning(['The best is to set iteration_number to negative values for optimal decision of iteration' newline ...
             'Reflection to the order n-1 are taken into account which may lead artifact (transmission n=1, single reflection n=2, higher n=?)'])
        h.Bornmax = h.iterations_number;
    end
    
    if h.verbose
        display(['number of step : ' num2str(h.Bornmax)])
        display(['step pixel size : ' num2str(round(step/h.resolution(3)))])
    end

    % phase ramp
    if h.acyclic
        if all(h.boundary_thickness_pixel(1:2)==0)
            x=exp(-1i.*pi.*((1:size(h.V,3))-1)./size(h.V,3)/2);
            x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
            h.phase_ramp={reshape(x,1,1,[])};
        else
            h.phase_ramp=cell(1,3);
            for dim = 1:3
                x=single(exp(-1i.*pi.*((1:size(h.V,dim))-1)./size(h.V,dim)/2));
                x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
                h.phase_ramp{dim}=reshape(x,circshift([1 1 length(x)], dim, 2));
            end
        end
    else
        h.phase_ramp=cell(0);
    end

    % Helmholtz Green function in Fourier space
    shifted_coordinate=h.utility.fourier_space.coor(1:3);
    if h.acyclic %shift all by kres/4 for acyclic convolution
        if ~h.cyclic_boundary_xy %shift only in z h.acyclic
            shifted_coordinate{1}=shifted_coordinate{1}+h.utility.fourier_space.res{1}/4;
            shifted_coordinate{2}=shifted_coordinate{2}+h.utility.fourier_space.res{2}/4;
        end
        shifted_coordinate{3}=shifted_coordinate{3}+h.utility.fourier_space.res{3}/4;
    end
    for i=1:3
        shifted_coordinate{i}=2*pi*ifftshift(gather(shifted_coordinate{i}));
    end

    if ~h.acyclic
        h.Greenp = xyz_periodic_green(h, shifted_coordinate);
    elseif h.cyclic_boundary_xy
        h.Greenp = xy_periodic_green(h, shifted_coordinate);
    else
        warning("Totally non-periodic Green's function is not yet implemented." + newline + ...
            "For now, xy-periodic green function is used.")
        h.Greenp = xy_periodic_green(h, shifted_coordinate);
    end
    
    h.flip_Greenp = fft_flip(h.Greenp,[1 1 1],false);
    if h.vector_simulation % dyadic Green's function
        eye_3=reshape(eye(3,'single'),1,1,1,3,3);
        rads=...
            shifted_coordinate{1}.*reshape([1 0 0],1,1,1,[])+...
            shifted_coordinate{2}.*reshape([0 1 0],1,1,1,[])+...
            shifted_coordinate{3}.*reshape([0 0 1],1,1,1,[]);
        green_absorbtion_correction=1/((2*pi*h.utility.k0_nm)^2+1i.*h.eps_imag);
        [xsize, ysize, zsize] = size(h.Greenp);
        flip_rads = fft_flip(rads,[1 1 1],false);
        h.Greenp = h.Greenp.*(eye_3-green_absorbtion_correction*(rads).*reshape(rads,xsize,ysize,zsize,1,3));
        h.flip_Greenp = h.flip_Greenp.*(eye_3-green_absorbtion_correction*(flip_rads).*reshape(flip_rads,xsize,ysize,zsize,1,3));
    end
    
    h.Greenp = single(gather(h.Greenp));
    h.flip_Greenp = single(gather(h.flip_Greenp));
end

function Greenp = xyz_periodic_green(h, shifted_coordinate)
    % Totally periodic Green's function
    Greenp = 1 ./ (abs(...
        (shifted_coordinate{1}).^2 + ...
        (shifted_coordinate{2}).^2 + ...
        (shifted_coordinate{3}).^2 ...
        )-(2*pi*h.utility.k0_nm)^2-1i*h.eps_imag);
end

function Greenp = xy_periodic_green(h, shifted_coordinate)
    % Z-axis-truncated Green's function
    % It is YX periodic
    Greenp = xyz_periodic_green(h, shifted_coordinate);
    k0 = sqrt((2*pi*h.utility.k0_nm)^2+1i*h.eps_imag - shifted_coordinate{2}.^2 - shifted_coordinate{3}.^2);
    kz = shifted_coordinate{3};
    L = (h.ROI(6)-h.ROI(5)+1)*h.resolution(3)/4;
    Greenp = Greenp .* (1- exp(1i*L*k0) .* (cos(kz*L) - 1i*kz./k0.*sin(kz*L)));
end