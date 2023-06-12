function set_kernel(obj)
    Nsize = size(obj.V);
    warning('off','all');
    obj.utility = derive_utility(obj, Nsize); % the utility for the space with border
    warning('on','all');
    
    % Maximum size of calculation of convergenent Born series
    if obj.iterations_number==0
        error('set iterations_number to either a positive or negative value');
    elseif obj.iterations_number<0
        step = abs(2*(2*pi*obj.utility.k0_nm)/obj.eps_imag);
        Bornmax_opt = ceil(norm(size(obj.V,1:3).*obj.resolution) / step / 2 + 1)*2;
        obj.Bornmax=Bornmax_opt*abs(obj.iterations_number);
    else
        warning(['The best is to set iteration_number to negative values for optimal decision of iteration' newline ...
             'Reflection to the order n-1 are taken into account which may lead artifact (transmission n=1, single reflection n=2, higher n=?)'])
        obj.Bornmax = obj.iterations_number;
    end
    
    if obj.verbose
        display(['number of step : ' num2str(obj.Bornmax)])
        display(['step pixel size : ' num2str(round(step/obj.resolution(3)))])
    end

    % phase ramp
    if obj.acyclic
        if all(obj.boundary_thickness_pixel(1:2)==0)
            x=exp(-1i.*pi.*((1:size(obj.V,3))-1)./size(obj.V,3)/2);
            x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
            obj.phase_ramp={reshape(x,1,1,[])};
        else
            obj.phase_ramp=cell(1,3);
            for dim = 1:3
                x=single(exp(-1i.*pi.*((1:size(obj.V,dim))-1)./size(obj.V,dim)/2));
                x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
                obj.phase_ramp{dim}=reshape(x,circshift([1 1 length(x)], dim, 2));
            end
        end
    else
        obj.phase_ramp=cell(0);
    end

    % Helmholtz Green function in Fourier space
    shifted_coordinate=obj.utility.fourier_space.coor(1:3);
    if obj.acyclic %shift all by kres/4 for acyclic convolution
        if ~obj.cyclic_boundary_xy %shift only in z obj.acyclic
            shifted_coordinate{1}=shifted_coordinate{1}+obj.utility.fourier_space.res{1}/4;
            shifted_coordinate{2}=shifted_coordinate{2}+obj.utility.fourier_space.res{2}/4;
        end
        shifted_coordinate{3}=shifted_coordinate{3}+obj.utility.fourier_space.res{3}/4;
    end
    for i=1:3
        shifted_coordinate{i}=2*pi*ifftshift(gather(shifted_coordinate{i}));
    end

    if ~obj.acyclic
        Greenp = xyz_periodic_green(obj, shifted_coordinate);
    elseif obj.cyclic_boundary_xy
        Greenp = xy_periodic_green(obj, shifted_coordinate);
    else
        warning("Totally non-periodic Green's function is not yet implemented." + newline + ...
            "For now, xy-periodic green function is used.")
        Greenp = xy_periodic_green(obj, shifted_coordinate);
    end
    if obj.use_GPU
        Greenp = gpuArray(Greenp);
    end
    
    flip_Greenp = fft_flip(Greenp,[1 1 1],false);
    if obj.vector_simulation % dyadic Green's function
        eye_3=reshape(eye(3,'single'),1,1,1,3,3);
        rads=...
            shifted_coordinate{1}.*reshape([1 0 0],1,1,1,[])+...
            shifted_coordinate{2}.*reshape([0 1 0],1,1,1,[])+...
            shifted_coordinate{3}.*reshape([0 0 1],1,1,1,[]);
        green_absorbtion_correction=1/((2*pi*obj.utility.k0_nm)^2+1i.*obj.eps_imag);
        [xsize, ysize, zsize] = size(Greenp);
        flip_rads = fft_flip(rads,[1 1 1],false);
        Greenp = Greenp.*(eye_3-green_absorbtion_correction*(rads).*reshape(rads,xsize,ysize,zsize,1,3));
        flip_Greenp = flip_Greenp.*(eye_3-green_absorbtion_correction*(flip_rads).*reshape(flip_rads,xsize,ysize,zsize,1,3));
    end
    if obj.vector_simulation
        obj.Green_fn = @(PSI, psi) apply_dyadic_Green(PSI, Greenp, psi);
        obj.flip_Green_fn = @(PSI, psi) apply_dyadic_Green(PSI, flip_Greenp, psi);
    else
        obj.Green_fn = @(PSI, psi) apply_scalar_Green(PSI, Greenp, psi);
        obj.flip_Green_fn = @(PSI, psi) apply_scalar_Green(PSI, flip_Greenp, psi);
    end
end

function PSI = apply_dyadic_Green(PSI, Greenp, psi)
    for i = 1:3
        psi(:,:,:,i) = fftn(psi(:,:,:,i));
    end
    for i = 1:3
        PSI =  PSI + Greenp(:,:,:,:,i).*psi(:,:,:,i);
    end
    for i = 1:3
        PSI(:,:,:,i) = ifftn(PSI(:,:,:,i));
    end
end

function PSI = apply_scalar_Green(PSI, Greenp, psi)
    psi = fftn(psi);
    PSI(:) = Greenp.*psi;
    PSI = ifftn(PSI);
end

function Greenp = xyz_periodic_green(obj, shifted_coordinate)
    % Totally periodic Green's function
    Greenp = 1 ./ (abs(...
        (shifted_coordinate{1}).^2 + ...
        (shifted_coordinate{2}).^2 + ...
        (shifted_coordinate{3}).^2 ...
        )-(2*pi*obj.utility.k0_nm)^2-1i*obj.eps_imag);
end

function Greenp = xy_periodic_green(obj, shifted_coordinate)
    % Z-axis-truncated Green's function
    % It is YX periodic
    Greenp = xyz_periodic_green(obj, shifted_coordinate);
    k0 = sqrt((2*pi*obj.utility.k0_nm)^2+1i*obj.eps_imag - shifted_coordinate{2}.^2 - shifted_coordinate{3}.^2);
    kz = shifted_coordinate{3};
    L = (obj.ROI(6)-obj.ROI(5)+1)*obj.resolution(3)/4;
    Greenp = Greenp .* (1- exp(1i*L*k0) .* (cos(kz*L) - 1i*kz./k0.*sin(kz*L)));
end