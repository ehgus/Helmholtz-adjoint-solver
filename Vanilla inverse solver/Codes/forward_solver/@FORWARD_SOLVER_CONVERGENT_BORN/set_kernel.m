function set_kernel(h)
    Nsize = size(h.RI);
    warning('off','all');
    h.utility = derive_utility(h, Nsize); % the utility for the space with border
    warning('on','all');
    
    if h.verbose && h.iterations_number>0
        warning('Best is to set iterations_number to -n for an automatic choice of this so that reflection to the order n-1 are taken in accound (transmission n=1, single reflection n=2, higher n=?)');
    end
    
    shifted_coordinate=h.utility.fourier_space.coor(1:3);
    if h.acyclic
        %shift all by kres/4
        if ~h.cyclic_boundary_xy
            %shift only in z h.acyclic
            shifted_coordinate{1}=shifted_coordinate{1}+h.utility.fourier_space.res{1}/4;
            shifted_coordinate{2}=shifted_coordinate{2}+h.utility.fourier_space.res{2}/4;
        end
        shifted_coordinate{3}=shifted_coordinate{3}+h.utility.fourier_space.res{3}/4;
    end
    for i=1:3
        shifted_coordinate{i}=ifftshift(shifted_coordinate{i});
    end
    
    if h.vector_simulation % need to make true k/4 shift for rad !!!
        rads=...
            (shifted_coordinate{1}./h.utility.k0_nm).*reshape([1 0 0],1,1,1,[])+...
            (shifted_coordinate{2}./h.utility.k0_nm).*reshape([0 1 0],1,1,1,[])+...
            (shifted_coordinate{3}./h.utility.k0_nm).*reshape([0 0 1],1,1,1,[]);
    end
    
    green_absorbtion_correction=((2*pi*h.utility.k0_nm)^2)/((2*pi*h.utility.k0_nm)^2+1i.*h.eps_imag);
    
    % Maximum size of calculation of convergenent Born series
    h.Bornmax = h.iterations_number;
    if h.iterations_number==0
        warning('set iterations_number to either a positive or negative value');
    elseif h.iterations_number<0
        step = abs(2*(2*pi*h.utility.k0_nm)/h.eps_imag);
        Bornmax_opt = ceil(norm(size(h.RI,1:3).*h.resolution) / step / 2 + 1)*2; % -CHANGED
        h.Bornmax=Bornmax_opt*abs(h.iterations_number);
    end
    
    if h.verbose
        display(['number of step : ' num2str(h.Bornmax)])
        display(['step pixel size : ' num2str(round(step/h.resolution(3)))])
    end
    
    eye_3=single(reshape(eye(3),1,1,1,3,3));

    % Helmholtz Green function in Fourier space
    h.Greenp = 1 ./ (4*pi^2.*abs(...
        (shifted_coordinate{1}).^2 + ...
        (shifted_coordinate{2}).^2 + ...
        (shifted_coordinate{3}).^2 ...
        )-(2*pi*h.utility.k0_nm)^2-1i*h.eps_imag);
    
    % phase ramp
    if h.acyclic
        if all(h.boundary_thickness_pixel(1:2)==0)
            x=exp(-1i.*pi.*((1:size(h.V,3))-1)./size(h.V,3)/2);
            x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
            h.phase_ramp=reshape(x,1,1,[]);
        else
            for j1 = 1:3
                x=single(exp(-1i.*pi.*((1:size(h.V,j1))-1)./size(h.V,j1)/2));
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
    if h.vector_simulation % dyadic Green's function
        [xsize, ysize, zsize] = size(h.Greenp);
        flip_rads = fft_flip(rads,[1 1 1],false);
        h.Greenp = h.Greenp.*(eye_3-green_absorbtion_correction*(rads).*reshape(rads,xsize,ysize,zsize,1,3));
        h.flip_Greenp = h.flip_Greenp.*(eye_3-green_absorbtion_correction*(flip_rads).*reshape(flip_rads,xsize,ysize,zsize,1,3));
    end
    
    h.Greenp = single(gather(h.Greenp));
    h.flip_Greenp = single(gather(h.flip_Greenp));
end