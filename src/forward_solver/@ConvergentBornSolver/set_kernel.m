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
    end

    % phase ramp
    if ~all(obj.periodic_boudnary)
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
    shifted_coordinate = gather(obj.utility.fourier_space.coor);
    flip_shifted_coordinate = gather(obj.utility.fourier_space.coor);
    for axis = 1:3
        shifted_coordinate{axis}=2*pi*ifftshift(shifted_coordinate{axis});
        flip_shifted_coordinate{axis}=2*pi*ifftshift(flip_shifted_coordinate{axis});
        if ~obj.periodic_boudnary(axis)
            shifted_coordinate{axis}=shifted_coordinate{axis}+2*pi*obj.utility.fourier_space.res{axis}/4;
            flip_shifted_coordinate{axis}=flip_shifted_coordinate{axis}-2*pi*obj.utility.fourier_space.res{axis}/4;
        end

    end
    k_square = (2*pi*obj.utility.k0_nm)^2+1i.*obj.eps_imag;

    if obj.use_GPU
        k_square = gpuArray(k_square);
        for axis = 1:3
            shifted_coordinate{axis}=gpuArray(shifted_coordinate{axis});
            flip_shifted_coordinate{axis}=gpuArray(flip_shifted_coordinate{axis});
        end
    end

    obj.Green_fn = DyadicGreen(k_square,shifted_coordinate{:});
    obj.flip_Green_fn = DyadicGreen(k_square,flip_shifted_coordinate{:});
end
