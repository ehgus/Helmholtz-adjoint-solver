function matt=padd_RI2conv(obj,matt)
    sz=size(matt);
    if ~obj.cyclic_boundary_xy
        size_conv=obj.size(1:2)'...
            +[obj.expected_RI_size(1) obj.expected_RI_size(2)]';
        
        add_start=-((floor(sz(1:2)'/2))-(floor(size_conv(:)/2)))...
            +[obj.RI_center(1) obj.RI_center(2)]';
        add_end=size_conv(:)-sz(1:2)'-add_start(:);
        
        matt=padarray(matt,add_start,0,'pre');
        matt=padarray(matt,add_end,0,'post');
    end
end