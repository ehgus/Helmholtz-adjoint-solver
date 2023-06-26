function matt=crop_field2RI(obj,matt)
    sz=size(matt);
    ROI_start=(floor(sz(1:2)'/2)+1)-(floor(obj.size(1:2)'/2))...
        +[obj.RI_center(1) obj.RI_center(2)]';
    ROI_end=ROI_start+obj.size(1:2)'-1;
    matt=matt(ROI_start(1):ROI_end(1),ROI_start(2):ROI_end(2),:,:);
end