classdef DyadicGreen
    properties
        use_GPU
        k_square
        k_idx
        phase_ramp
    end
    methods
        function obj=DyadicGreen(use_GPU, k_square,arr_size, resolution, subpixel_shift)
            obj.use_GPU=use_GPU;
            obj.k_square=k_square;
            freq_res = 2.*pi./(arr_size.*resolution);
            obj.k_idx = cell(1,3);
            obj.phase_ramp = cell(1,3);
            for axis = 1:3
                obj.k_idx{axis} = freq_res(axis).*reshape([0:ceil(arr_size(axis)/2-1) ceil(-arr_size(axis)/2):-1]+subpixel_shift(axis),circshift([1 1 arr_size(axis)],axis));
                if subpixel_shift(axis)~=0
                    obj.phase_ramp{axis} = single(exp(-2i.*pi.*reshape((0:arr_size(axis)-1)./arr_size(axis).*subpixel_shift(axis),circshift([1 1 arr_size(axis)],axis))));
                else
                    obj.phase_ramp{axis} = single(1);
                end
            end
            if obj.use_GPU
                obj.k_square = gpuArray(obj.k_square);
                for axis = 1:3
                    obj.k_idx{axis} = gpuArray(obj.k_idx{axis});
                    obj.phase_ramp{axis} = gpuArray(obj.phase_ramp{axis});
                end
            end
        end
        function dst = conv(obj,src,dst)
            if nargin == 2
                dst = zeros(size(src),'like',src);
            end
            % phase ramp
            if obj.use_GPU
                src(:) = arrayfun(@(a,x,y,z)a.*x.*y.*z,src,obj.phase_ramp{:});
            else
                if numel(obj.phase_ramp{1}) ~= 1
                    src(:) = src.*obj.phase_ramp{1};
                end
                if numel(obj.phase_ramp{2}) ~= 1
                    src(:) = src.*obj.phase_ramp{2};
                end
                if numel(obj.phase_ramp{3}) ~= 1
                    src(:) = src.*obj.phase_ramp{3};
                end
            end
            % frequency space
            for axis = 1:3
                src(:,:,:,axis) = fftn(src(:,:,:,axis));
            end
            % multiply dyadic Green's function
            if obj.use_GPU
                [dst(:,:,:,1),dst(:,:,:,2),dst(:,:,:,3)] = arrayfun(@multiply_Green,src(:,:,:,1),src(:,:,:,2),src(:,:,:,3),obj.k_idx{:},obj.k_square);
            else
                [dst(:,:,:,1),dst(:,:,:,2),dst(:,:,:,3)] = multiply_Green(src(:,:,:,1),src(:,:,:,2),src(:,:,:,3),obj.k_idx{:},obj.k_square);
            end
            % back to real space
            for axis = 1:3
                dst(:,:,:,axis) = ifftn(dst(:,:,:,axis));
            end
            % inverse phsae ramp
            if obj.use_GPU
                dst(:) = arrayfun(@(a,x,y,z)a.*conj(x.*y.*z),dst,obj.phase_ramp{:});
            else
                if numel(obj.phase_ramp{1}) ~= 1
                    dst(:) = dst./obj.phase_ramp{1};
                end
                if numel(obj.phase_ramp{2}) ~= 1
                    dst(:) = dst./obj.phase_ramp{2};
                end
                if numel(obj.phase_ramp{3}) ~= 1
                    dst(:) = dst./obj.phase_ramp{3};
                end
            end
        end
    end
end