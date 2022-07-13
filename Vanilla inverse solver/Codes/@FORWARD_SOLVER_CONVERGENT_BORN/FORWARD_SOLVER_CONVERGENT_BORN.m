classdef FORWARD_SOLVER_CONVERGENT_BORN < FORWARD_SOLVER
    properties %(SetAccess = protected, Hidden = true)
        utility_border;
        Bornmax;
        boundary_thickness_pixel;
        ROI;
        
        cyclic_boundary_xy;
        
        Greenp;
        rads;
        eye_3;
        
        V;
        pole_num;
        green_absorbtion_correction;
        eps_imag;
        
        kernel_trans;
        kernel_ref;
        
        attenuation_mask;
        pixel_step_size;
        
        phase_ramp;
        
        refocusing_util;
        
        expected_RI_size;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@FORWARD_SOLVER();
            %specific parameters
            params.iterations_number=-1;
            params.boundary_thickness = 6.*[1 1 1];
            params.boundary_sharpness = 1;%2;
            params.verbose = false;
            params.acyclic = true;
            params.RI_xy_size=[0 0];%if set to 0 the field is the same size as the simulation
            params.RI_center=[0 0];
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        [fields_trans,fields_ref,fields_3D]=solve(h,input_field);
        Field=solve_raw(h,source);
        Field=solve_adjoint(h,E_old,source)

        matt=padd_RI2conv(h,matt);
        matt=padd_field2conv(h,matt);
        matt=crop_conv2field(h,matt);
        matt=crop_conv2RI(h,matt);
        matt=crop_field2RI(h,matt);

        set_RI(h,RI);
        condition_RI(h);
        ROI = create_boundary_RI(h);
        init(h);

        function h=FORWARD_SOLVER_CONVERGENT_BORN(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            
            h@FORWARD_SOLVER(params);
            
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
                h.refocusing_util=gather(h.refocusing_util);
                h.refocusing_util= h.refocusing_util.*h.utility.NA_circle;
                free_space_green=h.refocusing_util./(1i*4*pi);
                free_space_green=free_space_green.*h.utility.NA_circle./(h.utility.k3+~h.utility.NA_circle);
                free_space_green=free_space_green./(h.utility.image_space.res{1}.*h.utility.image_space.res{2});
                free_space_green=fftshift(ifft2(ifftshift(free_space_green)));
            else
                params_truncated_green=h.parameters;
                params_truncated_green.size=h.parameters.size(:)...
                    +[h.expected_RI_size(1) h.expected_RI_size(2) 0]'...
                    +[h.parameters.RI_center(1) h.parameters.RI_center(2) 0]';
                
                warning('off','all');
                h.refocusing_util=(truncated_green_plus(params_truncated_green,true));
                h.refocusing_util=gather(h.refocusing_util);
                h.refocusing_util=h.refocusing_util(...
                    1-min(0,h.parameters.RI_center(1)):end-max(0,h.parameters.RI_center(1)),...
                    1-min(0,h.parameters.RI_center(2)):end-max(0,h.parameters.RI_center(2)),:);
                h.refocusing_util=circshift(h.refocusing_util,[-h.parameters.RI_center(1) -h.parameters.RI_center(2) 0]);
                h.refocusing_util=fftshift(ifftn(ifftshift(h.refocusing_util)));
                h.refocusing_util=fftshift(fft2(ifftshift(h.refocusing_util)));
                
                h.refocusing_util=h.refocusing_util.*(h.utility.image_space.res{1}.*h.utility.image_space.res{2});
                
                
                
                warning('off','all');
                free_space_green=(truncated_green_plus(params_truncated_green));
                warning('on','all');
                
                
                
                free_space_green=free_space_green(...
                    1-min(0,h.parameters.RI_center(1)):end-max(0,h.parameters.RI_center(1)),...
                    1-min(0,h.parameters.RI_center(2)):end-max(0,h.parameters.RI_center(2)),:);
                free_space_green=circshift(free_space_green,[-h.parameters.RI_center(1) -h.parameters.RI_center(2) 0]);
                free_space_green=fftshift(ifftn(ifftshift(free_space_green)));
                
                
            end
            
            h.kernel_trans=fftshift(fft2(ifftshift(conj(free_space_green))));
            h.kernel_ref=  fftshift(fft2(ifftshift((free_space_green))));
            
            h.kernel_ref=gather(h.kernel_ref);
            h.kernel_trans=gather(h.kernel_trans);
            
            warning('choose a higher size boundary to a size which fft is fast ??');
            warning('allow to chose a threshold for remaining energy');
            warning('min boundary size at low RI ??');

            if length(h.parameters.boundary_thickness) == 1
                error('Boundary in only one direction is not precise; enter "boundary_thickness" as an array of size 3 use size 0 for ciclic boundary');
            elseif length(h.parameters.boundary_thickness) == 3
                h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)./(h.parameters.resolution.*2));
            else
                error('h.boundary_thickness_pixel vector dimension should be 1 or 3.')
            end
        end



    end
end


