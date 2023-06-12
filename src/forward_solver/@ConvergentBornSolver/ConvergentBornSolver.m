classdef ConvergentBornSolver < ForwardSolver
    properties
        % scattering object w/ boundary
        V;
        expected_RI_size;
        % grid size
        RI_xy_size = [0 0 0];
        size;
        RI_center=[0 0];
        % boundary condition
        boundary_thickness = [0 0 0];
        boundary_thickness_pixel;
        field_attenuation = [0 0 0];
        field_attenuation_pixel;
        field_attenuation_sharpness = 1;
        field_attenuation_mask;
        potential_attenuation = [0 0 0];
        potential_attenuation_pixel;
        potential_attenuation_sharpness = 1;
        ROI;
        % CBS simulation option
        Bornmax;
        phase_ramp;
        cyclic_boundary_xy;
        acyclic logical = true;
        Green_fn;
        flip_Green_fn;
        iterations_number=-1;
        eps_imag = Inf;
        kernel_trans;
        kernel_ref;
        refocusing_util;
    end
    methods
        function h=ConvergentBornSolver(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h@ForwardSolver(params);
            % check boundary thickness
            if length(h.boundary_thickness) == 1
                h.boundary_thickness = ones(1,3) * h.boundary_thickness;
            end
            assert(length(h.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            h.boundary_thickness_pixel = double(round(h.boundary_thickness./(h.resolution.*2)));
            h.field_attenuation_pixel = double(round(h.field_attenuation./(h.resolution.*2)));
            h.potential_attenuation_pixel = double(round(h.potential_attenuation./(h.resolution.*2)));
            if h.RI_xy_size(1)==0
                h.RI_xy_size(1)=h.size(1);
            end
            if h.RI_xy_size(2)==0
                h.RI_xy_size(2)=h.size(2);
            end
            h.expected_RI_size=[h.RI_xy_size(1) h.RI_xy_size(2) h.size(3)];
            % set ROI
            h.ROI = [...
                h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+h.size(1)...
                h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+h.size(2)...
                h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+h.size(3)];
            % set field_attenuation_mask
            h.field_attenuation_mask=cell(0);
            for dim = 1:3
                max_L = h.boundary_thickness_pixel(dim);
                L = min(max_L, h.field_attenuation_pixel(dim));
                if h.boundary_thickness_pixel(dim)==0
                    continue;
                end
                window = (tanh(linspace(-2.5,2.5,L))/tanh(3)-tanh(-3))/2;
                window = window*h.field_attenuation_sharpness + (1-h.field_attenuation_sharpness);
                x = [window ones(1, h.ROI(2*(dim-1)+2) - h.ROI(2*(dim-1)+1) + 1 + 2*(max_L-L)) flip(window)];
                h.field_attenuation_mask{end+1} = reshape(x,circshift([1 1 length(x)],dim,2));
            end
            %make the cropped green function (for forward and backward field)
            sim_size = h.size + 2*h.boundary_thickness_pixel;
            h.utility = derive_utility(h, sim_size);
            h.cyclic_boundary_xy=(all(h.boundary_thickness(1:2)==0) && all(h.expected_RI_size(1:2)==h.size(1:2)));
            
            if h.cyclic_boundary_xy
                h.refocusing_util=exp(h.utility.refocusing_kernel.*h.utility.image_space.coor{3});
                h.refocusing_util=ifftshift(gather(h.refocusing_util));
                shifted_NA_circle = ifftshift(h.utility.fourier_space.coorxy  < h.utility.k0_nm);
                h.refocusing_util= h.refocusing_util.*shifted_NA_circle;
                free_space_green=h.refocusing_util/(4i*pi);
                free_space_green=free_space_green.*shifted_NA_circle./(ifftshift(h.utility.k3)+~shifted_NA_circle);
                free_space_green=free_space_green./(h.utility.image_space.res{1}*h.utility.image_space.res{2});
                free_space_green=ifft2(free_space_green);
            else
                params_truncated_green=struct( ...
                    'use_GPU', h.use_GPU, ...
                    'wavelength', h.wavelength, ...
                    'RI_bg', h.RI_bg, ...
                    'resolution', h.resolution, ...
                    'NA', h.NA, ...
                    'size', h.expected_RI_size(:) + [h.expected_RI_size(1) + h.RI_center(1), h.expected_RI_size(2) + h.RI_center(2), 0]' ...
                );
                warning('off','all');
                h.refocusing_util=truncated_green_plus(params_truncated_green,true);
                h.refocusing_util=gather(h.refocusing_util);
                h.refocusing_util=h.refocusing_util(...
                    1-min(0,h.RI_center(1)):end-max(0,h.RI_center(1)),...
                    1-min(0,h.RI_center(2)):end-max(0,h.RI_center(2)),:);
                h.refocusing_util=circshift(h.refocusing_util,[-h.RI_center(1) -h.RI_center(2) 0]);
                h.refocusing_util=ifft(ifftshift(h.refocusing_util),[],3);
                
                h.refocusing_util=h.refocusing_util*(h.utility.image_space.res{1}*h.utility.image_space.res{2});
                
                warning('off','all');
                free_space_green=truncated_green_plus(params_truncated_green);
                warning('on','all');
                
                free_space_green=free_space_green(...
                    1-min(0,h.RI_center(1)):end-max(0,h.RI_center(1)),...
                    1-min(0,h.RI_center(2)):end-max(0,h.RI_center(2)),:);
                free_space_green=circshift(free_space_green,[-h.RI_center(1) -h.RI_center(2) 0]);
                free_space_green=fftshift(ifftn(ifftshift(free_space_green)));
            end
            h.refocusing_util=fftshift(h.refocusing_util,3);
            free_space_green = free_space_green(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6));
            h.kernel_trans=gather(fft2(conj(free_space_green)));
            h.kernel_ref=  gather(fft2(free_space_green));
        end
    end
end


