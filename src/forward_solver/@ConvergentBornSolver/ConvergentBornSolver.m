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
        periodic_boudnary = [true true false];
        Green_fn;
        flip_Green_fn;
        iterations_number=-1;
        eps_imag = Inf;
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
        end
    end
end


