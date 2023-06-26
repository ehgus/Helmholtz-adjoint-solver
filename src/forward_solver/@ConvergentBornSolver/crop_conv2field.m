function matt=crop_conv2field(obj, matt)
    sz=size(matt);
    if ~obj.cyclic_boundary_xy
        ROI_start=(floor(sz(1:2)'/2)+1)-(floor(obj.size(1:2)'/2));
        ROI_end=ROI_start+obj.size(1:2)'-1;
        matt=matt(ROI_start(1):ROI_end(1),ROI_start(2):ROI_end(2),:,:);
    end
end