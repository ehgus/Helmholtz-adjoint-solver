function matt=crop_conv2RI(obj,matt)
    matt=obj.crop_conv2field(matt);
    matt=obj.crop_field2RI(matt);
end