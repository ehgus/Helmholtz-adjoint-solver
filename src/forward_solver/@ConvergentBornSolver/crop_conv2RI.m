function matt=crop_conv2RI(h,matt)
    matt=h.crop_conv2field(matt);
    matt=h.crop_field2RI(matt);
end