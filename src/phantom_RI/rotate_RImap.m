function new_RImap = rotate_RImap(RImap, rotation_angles)
    new_RImap = zeros(size(RImap),'single');
    RI_bg =  RImap(1); % assume edge of RI value as background
    RImap = imrotate3(RImap, rotation_angles(1)*180/pi, [0 0 1], 'crop','fillValues', RI_bg);
    RImap = imrotate3(RImap, rotation_angles(2)*180/pi, [1 0 0], 'crop','fillValues', RI_bg);
    RImap = imrotate3(RImap, rotation_angles(3)*180/pi, [0 0 1], 'crop','fillValues', RI_bg);
    new_RImap(floor(end/2)+1-floor(size(RImap,1)/2):floor(end/2) + size(RImap,1) - floor(size(RImap,1)/2),...
        floor(end/2)+1-floor(size(RImap,2)/2):floor(end/2) + size(RImap,2) - floor(size(RImap,2)/2),...
        floor(end/2)+1-floor(size(RImap,3)/2):floor(end/2) + size(RImap,3) - floor(size(RImap,3)/2)) = RImap;
end
