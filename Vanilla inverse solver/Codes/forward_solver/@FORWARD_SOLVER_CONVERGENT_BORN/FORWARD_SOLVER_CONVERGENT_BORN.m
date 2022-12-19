classdef FORWARD_SOLVER_CONVERGENT_BORN < FORWARD_SOLVER
    properties %(SetAccess = protected, Hidden = true)
        utility_border;
        Bornmax;
        boundary_thickness_pixel;
        ROI;
        
        cyclic_boundary_xy;
        
        Greenp;
        flip_Greenp;
        
        V;
        pole_num;
        green_absorbtion_correction;
        eps_imag = Inf;
        
        kernel_trans;
        kernel_ref;
        
        attenuation_mask;
        pixel_step_size;
        
        phase_ramp;
        
        refocusing_util;
        
        expected_RI_size;
    end
    methods
        function get_default_parameters(h)
            get_default_parameters@FORWARD_SOLVER(h);
            %specific parameters
            h.parameters.iterations_number=-1;
            h.parameters.boundary_thickness = [6 6 6];
            h.parameters.boundary_sharpness = 1;%2;
            h.parameters.verbose = false;
            h.parameters.acyclic = true;
            h.parameters.RI_xy_size=[0 0];%if set to 0 the field is the same size as the simulation
            h.parameters.RI_center=[0 0];
        end
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

        function h=FORWARD_SOLVER_CONVERGENT_BORN(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h@FORWARD_SOLVER(params);
            % check boundary thickness
            boundary_thickness = h.parameters.boundary_thickness;
            if length(boundary_thickness) == 1
                h.parameters.boundary_thickness = zeros(1,3);
                h.parameters.boundary_thickness(:) = boundary_thickness;
            end
            assert(length(h.parameters.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)./(h.parameters.resolution.*2));
            if h.parameters.RI_xy_size(1)==0
                h.parameters.RI_xy_size(1)=h.parameters.size(1);
            end
            if h.parameters.RI_xy_size(2)==0
                h.parameters.RI_xy_size(2)=h.parameters.size(2);
            end
            h.expected_RI_size=[h.parameters.RI_xy_size(1) h.parameters.RI_xy_size(2) h.parameters.size(3)];
            
            %make the cropped green function (for forward and backward field)
            h.cyclic_boundary_xy=(all(h.parameters.boundary_thickness(1:2)==0) && all(h.expected_RI_size(1:2)==h.parameters.size(1:2)));
            
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
                params_truncated_green=h.parameters;
                params_truncated_green.size=h.parameters.size(:)...
                    +[h.expected_RI_size(1) h.expected_RI_size(2) 0]'...
                    +[h.parameters.RI_center(1) h.parameters.RI_center(2) 0]';
                
                warning('off','all');
                h.refocusing_util=truncated_green_plus(params_truncated_green,true);
                h.refocusing_util=gather(h.refocusing_util);
                h.refocusing_util=h.refocusing_util(...
                    1-min(0,h.parameters.RI_center(1)):end-max(0,h.parameters.RI_center(1)),...
                    1-min(0,h.parameters.RI_center(2)):end-max(0,h.parameters.RI_center(2)),:);
                h.refocusing_util=circshift(h.refocusing_util,[-h.parameters.RI_center(1) -h.parameters.RI_center(2) 0]);
                h.refocusing_util=ifft(ifftshift(h.refocusing_util),[],3);
                
                h.refocusing_util=h.refocusing_util*(h.utility.image_space.res{1}*h.utility.image_space.res{2});
                
                warning('off','all');
                free_space_green=truncated_green_plus(params_truncated_green);
                warning('on','all');
                
                free_space_green=free_space_green(...
                    1-min(0,h.parameters.RI_center(1)):end-max(0,h.parameters.RI_center(1)),...
                    1-min(0,h.parameters.RI_center(2)):end-max(0,h.parameters.RI_center(2)),:);
                free_space_green=circshift(free_space_green,[-h.parameters.RI_center(1) -h.parameters.RI_center(2) 0]);
                free_space_green=fftshift(ifftn(ifftshift(free_space_green)));
            end
            h.kernel_trans=gather(fftshift(fft2(conj(free_space_green))));
            h.kernel_ref=  gather(fftshift(fft2(free_space_green)));
        end
        
    end
end


