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
    shifted_coordinate = obj.utility.fourier_space.coor(1:3);
    if obj.acyclic %shift all by kres/4 for acyclic convolution
        if ~obj.cyclic_boundary_xy %shift only in z obj.acyclic
            shifted_coordinate{1}=shifted_coordinate{1}+obj.utility.fourier_space.res{1}/4;
            shifted_coordinate{2}=shifted_coordinate{2}+obj.utility.fourier_space.res{2}/4;
        end
        shifted_coordinate{3}=shifted_coordinate{3}+obj.utility.fourier_space.res{3}/4;
    end
    for axis = 1:3
        shifted_coordinate{axis}=2*pi*ifftshift(gather(shifted_coordinate{axis}));
    end
    k_square = (2*pi*obj.utility.k0_nm)^2+1i.*obj.eps_imag;
    Lz = (obj.ROI(6)-obj.ROI(5)+1)*obj.resolution(3);
    if ~obj.acyclic
        Greenp = xyz_periodic_green(k_square, shifted_coordinate{:});
    elseif obj.cyclic_boundary_xy
        Greenp = xy_periodic_green(k_square, Lz, shifted_coordinate{:});
    else
        warning("Totally non-periodic Green's function is not yet implemented." + newline + ...
            "For now, xy-periodic green function is used.")
        Greenp = xy_periodic_green(k_square, Lz, shifted_coordinate{:});
    end
    
    flip_Greenp = fft_flip(Greenp,[1 1 1],false);
    % dyadic term
    rads=...
        shifted_coordinate{1}.*reshape([1 0 0],1,1,1,[])+...
        shifted_coordinate{2}.*reshape([0 1 0],1,1,1,[])+...
        shifted_coordinate{3}.*reshape([0 0 1],1,1,1,[]);
    flip_rads = fft_flip(rads, [1 1 1], false);
    rads = rads./sqrt(k_square);
    flip_rads = flip_rads./sqrt(k_square);

    if obj.use_GPU
        Greenp = gpuArray(Greenp);
        flip_Greenp = gpuArray(flip_Greenp);
        rads = gpuArray(rads);
        flip_rads = gpuArray(flip_rads);
    end

    obj.Green_fn = @(PSI, psi) apply_dyadic_Green(PSI, psi, Greenp, rads);
    obj.flip_Green_fn = @(PSI, psi) apply_dyadic_Green(PSI, psi, flip_Greenp, flip_rads);
end

function PSI = apply_dyadic_Green(PSI, psi, Greenp, rads)
    for axis = 1:3
        psi(:,:,:,axis) = fftn(psi(:,:,:,axis));
    end
    % identity term
    PSI(:) = Greenp.*psi;
    % dyadic term
    psi(:) = PSI.*rads;
    for axis = 1:3
        PSI(:) = PSI - rads.*psi(:,:,:,axis);
    end
    for axis = 1:3
        PSI(:,:,:,i) = ifftn(PSI(:,:,:,axis));
    end
end

function Greenp = xyz_periodic_green(k_square, kx, ky, kz)
    % Totally periodic Green's function
    Greenp = 1 ./ (abs(kx.^2 + ky.^2 + kz.^2)-k_square);
end

function Greenp = xy_periodic_green(k_square, Lz, kx, ky, kz)
    % Z-axis-truncated Green's function
    % It is YX periodic
    Greenp = xyz_periodic_green(k_square, kx, ky, kz);
    k0 = sqrt(k_square - kx.^2 - ky.^2);
    Greenp = Greenp .* (1- exp(1i*Lz*k0) .* (cos(kz*Lz) - 1i*kz./k0.*sin(kz*Lz)));
end