classdef(Abstract) CurrentSource < OpticalSimulation
    properties
        outcoming_wave = true
        polarization
        grid_size
        RI_bg
    end
    methods
        function obj = CurrentSource(options)
            obj@OpticalSimulation(options);
        end
    end
    methods(Abstract)
        Efield = generate_Efield(obj, padding_size) % padding_size = array size of (2,3)
        Hfield = generate_Hfield(obj, padding_size)
    end
end