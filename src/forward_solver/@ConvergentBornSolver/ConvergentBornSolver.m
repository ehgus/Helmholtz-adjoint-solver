classdef ConvergentBornSolver < ForwardSolver
    properties
        % scattering object w/ boundary
        V;
        expected_RI_size;
        boundary_thickness_pixel;
        boundary_thickness = [6 6 6];
        boundary_sharpness = 1;
        max_attenuation_width = [0 0 0];
        max_attenuation_width_pixel;
        RI_xy_size = [0 0 0];
        size;
        RI_center=[0 0];
        attenuation_mask;
        phase_ramp;
        ROI;
        Bornmax;
        % FDTD option
        cyclic_boundary_xy;
        acyclic logical = true;
        Greenp;
        flip_Greenp;
        iterations_number=-1;
        eps_imag = Inf;
        kernel_trans;
        kernel_ref;
        refocusing_util;
    end
    methods
        [fields_trans,fields_ref,fields_3D]=solve(h,input_field);
        Field=solve_forward(h,source);

        matt=padd_RI2conv(h,matt);
        matt=padd_field2conv(h,matt);
        matt=crop_conv2field(h,matt);
        matt=crop_conv2RI(h,matt);
        matt=crop_field2RI(h,matt);

        set_RI(h,RI);
        condition_RI(h);
        create_boundary_RI(h);
        set_kernel(h);

        function h=ConvergentBornSolver(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h@ForwardSolver(params);
            % check boundary thickness
            boundary_thickness = h.boundary_thickness;
            attenuation_width = h.max_attenuation_width;
            if length(boundary_thickness) == 1
                h.boundary_thickness = ones(1,3) * boundary_thickness;
            end
            assert(length(h.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            h.boundary_thickness_pixel = double(round((h.boundary_thickness*h.wavelength/abs(h.RI_bg))./(h.resolution.*2)));
            h.max_attenuation_width_pixel = double(round((attenuation_width*h.wavelength/abs(h.RI_bg))./(h.resolution.*2)));
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
            %make the cropped green function (for forward and backward field)
            sim_size = h.size + 2*h.boundary_thickness_pixel;
            h.utility = derive_utility(h, sim_size);
            h.cyclic_boundary_xy=(all(h.boundary_thickness(1:2)==0) && all(h.expected_RI_size(1:2)==h.size(1:2)));
            
            if h.cyclic_boundary_xy
                h.refocusing_util=exp(h.utility.refocusing_kernel.*h.utility.image_space.coor{3});
                h.refocusing_util=ifftshift(gather(h.refocusing_util));
                shifted_NA_circle = ifftshift(h.utility.NA_circle);
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
            free_space_green = free_space_green(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6));
            h.kernel_trans=gather(fft2(conj(free_space_green)));
            h.kernel_ref=  gather(fft2(free_space_green));
        end
        
    end
end


