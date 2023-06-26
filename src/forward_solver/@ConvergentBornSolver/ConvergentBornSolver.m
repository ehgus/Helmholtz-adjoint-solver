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
        function obj=ConvergentBornSolver(params)
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            obj@ForwardSolver(params);
            % check boundary thickness
            if length(obj.boundary_thickness) == 1
                obj.boundary_thickness = ones(1,3) * obj.boundary_thickness;
            end
            assert(length(obj.boundary_thickness) == 3, 'boundary_thickness should be either a 3-size vector or a scalar')
            obj.boundary_thickness_pixel = double(round(obj.boundary_thickness./(obj.resolution.*2)));
            obj.field_attenuation_pixel = double(round(obj.field_attenuation./(obj.resolution.*2)));
            obj.potential_attenuation_pixel = double(round(obj.potential_attenuation./(obj.resolution.*2)));
            if obj.RI_xy_size(1)==0
                obj.RI_xy_size(1)=obj.size(1);
            end
            if obj.RI_xy_size(2)==0
                obj.RI_xy_size(2)=obj.size(2);
            end
            obj.expected_RI_size=[obj.RI_xy_size(1) obj.RI_xy_size(2) obj.size(3)];
            % set ROI
            obj.ROI = [...
                obj.boundary_thickness_pixel(1)+1 obj.boundary_thickness_pixel(1)+obj.size(1)...
                obj.boundary_thickness_pixel(2)+1 obj.boundary_thickness_pixel(2)+obj.size(2)...
                obj.boundary_thickness_pixel(3)+1 obj.boundary_thickness_pixel(3)+obj.size(3)];
            % set field_attenuation_mask
            obj.field_attenuation_mask=cell(0);
            for dim = 1:3
                max_L = obj.boundary_thickness_pixel(dim);
                L = min(max_L, obj.field_attenuation_pixel(dim));
                if obj.boundary_thickness_pixel(dim)==0
                    continue;
                end
                window = (tanh(linspace(-2.5,2.5,L))/tanh(3)-tanh(-3))/2;
                window = window*obj.field_attenuation_sharpness + (1-obj.field_attenuation_sharpness);
                x = [window ones(1, obj.ROI(2*(dim-1)+2) - obj.ROI(2*(dim-1)+1) + 1 + 2*(max_L-L)) flip(window)];
                obj.field_attenuation_mask{end+1} = reshape(x,circshift([1 1 length(x)],dim,2));
            end
            %make the cropped green function (for forward and backward field)
            sim_size = obj.size + 2*obj.boundary_thickness_pixel;
            obj.utility = derive_utility(obj, sim_size);
            obj.cyclic_boundary_xy=(all(obj.boundary_thickness(1:2)==0) && all(obj.expected_RI_size(1:2)==obj.size(1:2)));
            
            if obj.cyclic_boundary_xy
                obj.refocusing_util=exp(obj.utility.refocusing_kernel.*obj.utility.image_space.coor{3});
                obj.refocusing_util=ifftshift(gather(obj.refocusing_util));
                shifted_NA_circle = ifftshift(obj.utility.fourier_space.coorxy  < obj.utility.k0_nm);
                obj.refocusing_util= obj.refocusing_util.*shifted_NA_circle;
                free_space_green=obj.refocusing_util/(4i*pi);
                free_space_green=free_space_green.*shifted_NA_circle./(ifftshift(obj.utility.k3)+~shifted_NA_circle);
                free_space_green=free_space_green./(obj.utility.image_space.res{1}*obj.utility.image_space.res{2});
                free_space_green=ifft2(free_space_green);
            else
                params_truncated_green=struct( ...
                    'use_GPU', obj.use_GPU, ...
                    'wavelength', obj.wavelength, ...
                    'RI_bg', obj.RI_bg, ...
                    'resolution', obj.resolution, ...
                    'NA', obj.NA, ...
                    'size', obj.expected_RI_size(:) + [obj.expected_RI_size(1) + obj.RI_center(1), obj.expected_RI_size(2) + obj.RI_center(2), 0]' ...
                );
                warning('off','all');
                obj.refocusing_util=truncated_green_plus(params_truncated_green,true);
                obj.refocusing_util=gather(obj.refocusing_util);
                obj.refocusing_util=obj.refocusing_util(...
                    1-min(0,obj.RI_center(1)):end-max(0,obj.RI_center(1)),...
                    1-min(0,obj.RI_center(2)):end-max(0,obj.RI_center(2)),:);
                obj.refocusing_util=circshift(obj.refocusing_util,[-obj.RI_center(1) -obj.RI_center(2) 0]);
                obj.refocusing_util=ifft(ifftshift(obj.refocusing_util),[],3);
                
                obj.refocusing_util=obj.refocusing_util*(obj.utility.image_space.res{1}*obj.utility.image_space.res{2});
                
                warning('off','all');
                free_space_green=truncated_green_plus(params_truncated_green);
                warning('on','all');
                
                free_space_green=free_space_green(...
                    1-min(0,obj.RI_center(1)):end-max(0,obj.RI_center(1)),...
                    1-min(0,obj.RI_center(2)):end-max(0,obj.RI_center(2)),:);
                free_space_green=circshift(free_space_green,[-obj.RI_center(1) -obj.RI_center(2) 0]);
                free_space_green=fftshift(ifftn(ifftshift(free_space_green)));
            end
            obj.refocusing_util=fftshift(obj.refocusing_util,3);
        end
    end
end


