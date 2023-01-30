function RImap = phantom_plate(Nsize, RI_profile, thickness_pixel)
    % Return RI map of n phantom beads with specific pixel radius
    % Nsize: size of RI map
    % RI_profile (n-elements vector): RI of each plates. The last plate will fill the remainder
    % thickness_pixel ((n-1)-elements vector): thickness in pixel. It should be incremental order
    RImap = zeros(Nsize,'single');
    RImap(:) = RI_profile(end);
    end_idx = 1;
    for idx = 1:length(RI_profile)-1
        RImap(:,:,end_idx:end) = RI_profile(idx);
        end_idx = end_idx + thickness_pixel(idx);
    end
    RImap(:,:,end_idx:end) = RI_profile(end); 
end