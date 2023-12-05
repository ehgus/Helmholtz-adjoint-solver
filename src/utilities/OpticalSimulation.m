classdef (Abstract) OpticalSimulation < handle
    % Basic optical simulation class
    % The class contains required parameters to simulate linear system under coherent light source
    % It also provides basic instance initializer
    properties
        % field information
        wavelength {mustBePositive} = 1;    %wavelength [um]
        % scattering object information: set_RI define these properties
        RI = 1;                             %Refractive index map: size(RI, 1:3) = number of grids for each axis, size(RI, 4:5) = if isotropic [1, 1] else [3, 3]
        impedance = 120*pi;                 %wave impedance: size(RI, 1:3) = number of grids for each axis, size(RI, 4:5) = if isotropic [1, 1] else [3, 3]
        resolution(1,3) = [1 1 1];          %3D Voxel size [um]
        % configuration
        verbose = false;                    %verbose option for narrative report
        verbose_level = 0;                  %degree of verbosity
    end
    methods
        function obj = OpticalSimulation(options)
            obj = obj.update_parameters(options);
        end
        
        function obj=update_parameters(obj, options)
            % Update parameters from default settings
            % It ignores unsupported fields
            instance_properties = intersect(properties(obj),fieldnames(options));
            for idx = 1:length(instance_properties)
                property = instance_properties{idx};
                obj.(property) = options.(property);
            end
        end
    end
end