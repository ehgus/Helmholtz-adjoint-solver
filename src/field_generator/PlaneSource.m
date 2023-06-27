classdef PlaneSource < CurrentSource
    properties
       center_position  % center of the field (grid position)
       direction = 3    % polarization axis: 1 = 'X', 2 = 'Y', 3 = 'Z'
       horizontal_k_vector
    end
    methods
        function obj = PlaneSource(options)
            obj@CurrentSource(options);
            % This function does not check the orthogonality of k-vector and polarizaiton
        end
        function Efield = generate_Efield(obj, padding_size)
            unit_phase = zeros(1,3);
            % axial phase
            k_axial = sqrt((2*pi/obj.wavelength*obj.RI_bg)^2 - sum(obj.horizontal_k_vector.^2));
            unit_phase(obj.direction) = k_axial * obj.resolution(obj.direction);
            % horizontal phase
            horizontal_idx = circshift(1:3,-obj.direction);
            horizontal_idx = horizontal_idx(1:2);
            unit_phase(horizontal_idx) = obj.horizontal_k_vector .* obj.resolution(horizontal_idx);
            % generate field
            field_polarization = reshape(obj.polarization, 1, 1, 1, 3);
            ranges = { ...`
            reshape(1-padding_size(1, 1):obj.grid_size(1)+padding_size(2,1), [], 1) - obj.center_position(1), ...
            reshape(1-padding_size(1, 2):obj.grid_size(2)+padding_size(2,2), 1, []) - obj.center_position(2), ...
            reshape(1-padding_size(1, 3):obj.grid_size(3)+padding_size(2,3), 1, 1, []) - obj.center_position(3) ...
            };
            ranges{obj.direction} = abs(ranges{obj.direction});
            phase_map = exp(1i .* (ranges{1}.*unit_phase(1) + ranges{2}.*unit_phase(2) + ranges{3}.*unit_phase(3)));
            Efield = bsxfun(@times, phase_map, field_polarization);
        end
        function Hfield = generate_Hfield(obj, padding_size)
            unit_phase = zeros(1,3);
            k_vector = zeros(1,3);
            % axial phase
            k_axial = sqrt((2*pi/obj.wavelength*obj.RI_bg)^2 - sum(obj.horizontal_k_vector.^2));
            unit_phase(obj.direction) = k_axial * obj.resolution(obj.direction);
            k_vector(obj.direction) = k_axial;
            % horizontal phase
            horizontal_idx = circshift(1:3,-obj.direction);
            horizontal_idx = horizontal_idx(1:2);
            unit_phase(horizontal_idx) = obj.horizontal_k_vector .* obj.resolution(horizontal_idx);
            k_vector(horizontal_idx) = obj.horizontal_k_vector;
            % generate field
            impedance = 377/obj.RI_bg;
            unit_k_vector = k_vector./sqrt(sum(abs(k_vector).^2));
            field_polarization = cross(unit_k_vector, obj.polarization)/impedance;
            field_polarization = reshape(field_polarization,1,1,1,3);
            ranges = { ...
            reshape(1-padding_size(1, 1):obj.grid_size(1)+padding_size(2,1), [], 1) - obj.center_position(1), ...
            reshape(1-padding_size(1, 2):obj.grid_size(2)+padding_size(2,2), 1, []) - obj.center_position(2), ...
            reshape(1-padding_size(1, 3):obj.grid_size(3)+padding_size(2,3), 1, 1, []) - obj.center_position(3) ...
            };
            ranges{4} = sign(ranges{obj.direction});
            ranges{obj.direction} = abs(ranges{obj.direction});
            phase_map = exp(1i .* (ranges{1}.*unit_phase(1) + ranges{2}.*unit_phase(2) + ranges{3}.*unit_phase(3)));
            Hfield = bsxfun(@times, phase_map, field_polarization);
            Hfield = Hfield .* ranges{4};
        end
    end
end