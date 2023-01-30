function matt=padd_field2conv(h,matt)
    sz=size(matt);
    if ~h.cyclic_boundary_xy
        size_conv=h.size(1:2)'...
            +[h.expected_RI_size(1) h.expected_RI_size(2)]';
        
        add_start=-((floor(sz(1:2)'/2))-(floor(size_conv(:)/2)));
        add_end=size_conv(:)-sz(1:2)'-add_start(:);
        
        matt=padarray(matt,add_start,0,'pre');
        matt=padarray(matt,add_end,0,'post');
    end
end