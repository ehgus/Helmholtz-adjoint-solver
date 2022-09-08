classdef FORWARD_SOLVER_CONVERGENT_BORN < FORWARD_SOLVER
    properties
        iterations_number=-1;
        boundary_thickness = [6 6 6];
        boundary_sharpness = 1;
        acyclic = true;
        RI_xy_size = [0 0];
        RI_center = [0 0];

        Bornmax;
        boundary_thickness_pixel;
        ROI;
        
        cyclic_boundary_xy;
        
        Greenp;
        flip_Greenp;
        
        potential;
        pole_num;
        green_absorbtion_correction;
        eps_imag = Inf;
        
        return_transmission=false;  %return transmission field
        return_reflection=false;    %return reflection field
        return_3D=true;             %return 3D field
        kernel_trans;
        kernel_ref;
        
        attenuation_mask;
        pixel_step_size;
        
        phase_ramp;
        
        refocusing_util;
        
        expected_RI_size;
    end
    methods
        [fields_trans,fields_ref,fields_3D]=solve(h,input_field);
        Field=solve_forward(h,source);

        mat=padd_RI2conv(h,mat);
        mat=padd_field2conv(h,mat);
        mat=crop_conv2field(h,mat);
        mat=crop_conv2RI(h,mat);
        mat=crop_field2RI(h,mat);

        set_RI(h,RI);
        condition_RI(h);
        create_boundary_RI(h);
        set_kernel(h);

        function h=FORWARD_SOLVER_CONVERGENT_BORN(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h = h@FORWARD_SOLVER(params)
            assert(length(h.boundary_thickness) == 3, 'h.boundary_thickness_pixel vector dimension should be 3.')
            
            if h.RI_xy_size(1)==0
                h.RI_xy_size(1)=h.size(1);
            end
            if h.RI_xy_size(2)==0
                h.RI_xy_size(2)=h.size(2);
            end
            h.expected_RI_size=[h.RI_xy_size(1) h.RI_xy_size(2) h.size(3)];
            
            %make the cropped green function (for forward and backward field)
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
                params_truncated_green=h;
                params_truncated_green.size=h.size(:)...
                    +[h.expected_RI_size(1) h.expected_RI_size(2) 0]'...
                    +[h.RI_center(1) h.RI_center(2) 0]';
                
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
            
            h.kernel_trans=fftshift(fft2(conj(free_space_green)));
            h.kernel_ref=  fftshift(fft2(free_space_green));
            
            h.kernel_ref=gather(h.kernel_ref);
            h.kernel_trans=gather(h.kernel_trans);

            h.boundary_thickness_pixel = round((h.boundary_thickness*h.wavelength/h.RI_bg)./(h.resolution.*2));
        end
        
    end
end


