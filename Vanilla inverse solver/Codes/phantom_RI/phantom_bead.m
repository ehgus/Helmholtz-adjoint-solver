function RImap = phantom_bead(Nsize, RI_profile, pixel_radius, num_bead, bead_distance)
    % Return RI map of n phantom beads with specific pixel radius
    % Nsize: size of RI map
    % RI_profile: RI of background (first element) and beads (second element)
    % pixel_radius: radius of bead in pixel
    % num_bead( = 1): number of beads 
    % bead_distance( = 2* pixel_radius): distance between the nearest beads from center
    narginchk(3, 5)
    if nargin == 3
        num_bead = 1;
    elseif nargin == 4
        bead_distance = 2*pixel_radius;
    else
        assert(bead_distance >= 2*pixel_radius, "The distance between beads should be larger than its radius")
    end
    RI_bg = RI_profile(1);
    RI_sp = RI_profile(2);
    RI_delta = RI_sp - RI_bg;
    RImap = zeros(Nsize,'single');
    axis = arrayfun(@(dim_size) -floor(dim_size/2):ceil(dim_size/2)-1, Nsize,'UniformOutput',false);
    for aliasing_number = 0:7
        x_shift = 1/4;y_shift=1/4;z_shift=1/4;
        if floor(aliasing_number/4) == 1
            x_shift = -1/4;
        end
        if floor(rem(aliasing_number, 4)/2) == 1
            y_shift = -1/4;
        end
        if rem(aliasing_number,2) == 1
            z_shift = -1/4;
        end
        r_norm = sqrt(reshape(axis{1} + x_shift,[],1).^2 + reshape(axis{2} + y_shift,1,[]).^2 + reshape(axis{3} + z_shift,1,1,[]).^2);
        bead_mask = r_norm < pixel_radius;
        if num_bead == 1
            % single bead
            RImap(bead_mask) = RImap(bead_mask) + RI_delta;
        else
            for bead_idx = 1:num_bead
                angle = 2*pi/num_bead*(bead_idx-1);
                bead_center = round(bead_distance/2*[cos(angle) sin(angle) 0]);
                RImap(circshift(bead_mask, bead_center)) = RImap(circshift(bead_mask, bead_center)) + RI_delta;
            end
        end
    end
    RImap = RImap / 8;
    RImap= RImap + RI_bg;
end