classdef (Abstract) ForwardSolver < OpticalSimulation
    properties
        % Additional field information
        NA {mustBePositive} = 1;            %Numerical aperture of input/output waves
        use_abbe_sine logical = true;       %Abbe sine condition according to demagnification condition
        utility
        % Addtional scattering object information: set_RI define these properties
        RI_bg;                              %The representative refractive index
        % acceleration
        use_GPU logical = true;             %GPU acceleration
    end
    methods(Abstract)
        solve(obj, input_field)
        set_RI(obj, RI) % determine RI and RI_bg
    end
    methods
        function obj=ForwardSolver(options)
            obj@OpticalSimulation(options);
        end
        function utility = derive_utility(obj, Nsize)
            params = struct( ...
                'resolution', obj.resolution ,...
                'size', Nsize, ...
                'wavelength', obj.wavelength, ...
                'RI_bg', obj.RI_bg, ...
                'NA', obj.NA ...
            );
            utility = derive_optical_tool(params, obj.use_GPU);
        end
    end
end
