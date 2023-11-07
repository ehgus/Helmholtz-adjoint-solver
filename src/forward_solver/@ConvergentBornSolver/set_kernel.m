function set_kernel(obj)
    k0_nm = 2*pi*obj.RI_bg/obj.wavelength;
    % Maximum size of calculation of convergenent Born series
    if obj.iterations_number==0
        error('set iterations_number to either a positive or negative value');
    elseif obj.iterations_number<0
        step = abs(2*k0_nm/obj.eps_imag);
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

    % Helmholtz Green function in Fourier space
    arr_size = size(obj.V);
    resolution = obj.resolution;
    subpixel_shift = [1/4 1/4 1/4].*(~obj.periodic_boudnary);
    flip_subpixel_shift = -subpixel_shift;
    k_square = k0_nm^2+1i.*obj.eps_imag;

    obj.Green_fn = DyadicGreen(obj.use_GPU,k_square,arr_size,resolution,subpixel_shift);
    obj.flip_Green_fn = DyadicGreen(obj.use_GPU,k_square,arr_size,resolution,flip_subpixel_shift);
end
