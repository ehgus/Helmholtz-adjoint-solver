classdef OPTICAL_SIMULATION < handle
    % Basic optical simulation class
    properties
        parameters;
    end
    methods
        function h = OPTICAL_SIMULATION(params)
            h.get_default_parameters();
            h.update_properties(params);
        end

        function get_default_parameters(h)
            % Set default parameters for simulation
            h.parameters = struct;
            h.parameters.size=[100 100 100];      %3D volume grid
            h.parameters.wavelength=0.532;        %wavelength [um]
            h.parameters.NA=1.2;                  %objective lens NA
            h.parameters.RI_bg=1.336;             %Background RI
            h.parameters.resolution=[0.1 0.1 0.1];%3D Voxel size [um]
            h.parameters.vector_simulation=true;  %True/false: dyadic/scalar Green's function
            h.parameters.use_abbe_sine=true;      %Abbe sine condition according to demagnification condition
        end
        
        function h = update_properties(h, params)
            % Update parameters from default settings
            % It ignores unsupported fields.
            fields=intersect(fieldnames(params),fieldnames(h.parameters));
            for i = 1:length(fields)
                field=fields{i};
                h.parameters.(field) = params.(field);
            end
        end
    end
end