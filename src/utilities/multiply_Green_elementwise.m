function [dst_x,dst_y,dst_z] = multiply_Green_elementwise(src_x, src_y, src_z, kx, ky, kz, k_square)
    % multiply dyadic term
    % note: `arrayfun` only support element-wise multiplication
    dst_x = (kx.*src_x + ky.*src_y + kz.*src_z)./k_square;
    
    dst_z = kz.*dst_x;
    dst_y = ky.*dst_x;
    dst_x = kx.*dst_x;

    dst_x = src_x - dst_x;
    dst_y = src_y - dst_y;
    dst_z = src_z - dst_z;
    % multiply scalar Green's function
    Green_fn = 1 ./ (abs(kx.^2 + ky.^2 + kz.^2)-k_square);

    dst_x = dst_x.*Green_fn;
    dst_y = dst_y.*Green_fn;
    dst_z = dst_z.*Green_fn;
end