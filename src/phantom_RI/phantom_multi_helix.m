function RImap = phantom_multi_helix(Nsize, RI_profile, pixel_radius, one_turn_length, helix_distance, num_helix)
    % Return RI map of n phantom beads with specific pixel radius
    % Nsize: size of RI map
    % RI_profile: RI of background (first element) and beads (second element)
    % pixel_radius: radius of bead in pixel
    % one_turn_length: axial length for one turn
    % helix_distance: distance between two helix structure
    narginchk(5,6)
    if nargin == 5
        num_helix = 1;
    end
    RI_bg = RI_profile(1);
    RI_sp = RI_profile(2);
    RImap = zeros(Nsize, 'like', RI_profile);
    RImap(:) = RI_bg;
    
    axis = arrayfun(@(dim_size) - floor(dim_size/2):ceil(dim_size/2)-1, Nsize, 'UniformOutput',false);
    for angle = linspace(2*pi/num_helix,2*pi, num_helix)
        center_pos = helix_distance./2.*[cos(2*pi*axis{3}./one_turn_length + angle); sin(2*pi*axis{3}./one_turn_length + angle)];
        center_pos = reshape(center_pos, 2, 1, []);
        helix_mask = (reshape(axis{1}, [], 1) - center_pos(1,1,:)).^2 + (reshape(axis{2}, 1, []) - center_pos(2, 1, :)).^2 < pixel_radius^2;
        RImap(helix_mask) = RI_sp;
    end
end