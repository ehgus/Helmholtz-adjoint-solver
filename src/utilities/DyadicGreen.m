classdef DyadicGreen
    properties
        k_square
        kx
        ky
        kz
    end
    methods
        function obj=DyadicGreen(k_square, kx, ky, kz)
            obj.k_square=k_square;
            obj.kx = kx;
            obj.ky = ky;
            obj.kz = kz;
        end
        function dst = conv(obj,src,dst)
            if nargin == 2
                dst = zeros(size(src),'like',src);
            end
            % frequency space
            for axis = 1:3
                src(:,:,:,axis) = fftn(src(:,:,:,axis));
            end
            % multiply dyadic Green's function
            if isgpuarray(src)
                [dst(:,:,:,1),dst(:,:,:,2),dst(:,:,:,3)] = arrayfun(@multiply_Green,src(:,:,:,1),src(:,:,:,2),src(:,:,:,3),obj.kx,obj.ky,obj.kz,obj.k_square);
            else
                [dst(:,:,:,1),dst(:,:,:,2),dst(:,:,:,3)] = multiply_Green(src(:,:,:,1),src(:,:,:,2),src(:,:,:,3),obj.kx,obj.ky,obj.kz,obj.k_square);
            end
            % back to real space
            for axis = 1:3
                dst(:,:,:,axis) = ifftn(dst(:,:,:,axis));
            end
        end
    end
end