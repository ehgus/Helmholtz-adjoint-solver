function matt=crop_field2RI(h,matt)
    sz=size(matt);
    ROI_start=(floor(sz(1:2)'/2)+1)-(floor(h.size(1:2)'/2))...
        +[h.RI_center(1) h.RI_center(2)]';
    ROI_end=ROI_start+h.size(1:2)'-1;
    matt=matt(ROI_start(1):ROI_end(1),ROI_start(2):ROI_end(2),:,:);
end