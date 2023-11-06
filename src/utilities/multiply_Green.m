function [dst_x,dst_y,dst_z] = multiply_Green(src_x, src_y, src_z, kx, ky, kz, k_square)
    % multiply dyadic term
    % note: `arrayfun` only support element-wise multiplication
    kxx = 1-kx.*kx./k_square;
    kyy = 1-ky.*ky./k_square;
    kzz = 1-kz.*kz./k_square;
    kxy =  -kx.*ky./k_square;
    kxz =  -kx.*kz./k_square;
    kyz =  -ky.*kz./k_square;

    dst_x = kxx.*src_x + kxy.*src_y + kxz.*src_z;
    dst_y = kxy.*src_x + kyy.*src_y + kyz.*src_z;
    dst_z = kxz.*src_x + kyz.*src_y + kzz.*src_z;
    % multiply scalar Green's function
    Green_fn = 1 ./ (abs(kx.^2 + ky.^2 + kz.^2)-k_square);

    dst_x = dst_x.*Green_fn;
    dst_y = dst_y.*Green_fn;
    dst_z = dst_z.*Green_fn;
end