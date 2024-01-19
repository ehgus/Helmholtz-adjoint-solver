classdef CyclicConv2Regularizer < Regularizer
    % See 'Inverse Design of Nanophotonic Devices with Structural Integrity'
    % or 'Photonic topology optimization with semiconductor-foundry design-rule constraints'
    properties
        kernel
        slice_axis
    end
    properties(Hidden)
        arr_slice
        kernel_sum
    end
    methods (Static)
        function kernel = gaussian(pixel_radius, sigma)
            arguments
                pixel_radius {mustBePositive,mustBeInteger}
                sigma {mustBeReal}
            end
            coord = -pixel_radius:pixel_radius;
            kernel = exp(-(coord.^2 + coord'.^2)./(2*sigma^2));
            kernel(coord.^2 + coord'.^2 > pixel_radius.^2) = 0;
        end
        function kernel = conic(pixel_radius)
            arguments
                pixel_radius {mustBePositive,mustBeInteger}
            end
            coord = -pixel_radius:pixel_radius;
            kernel = 1-sqrt(coord.^2 + coord'.^2)./pixel_radius;
            kernel(coord.^2 + coord'.^2 > pixel_radius.^2) = 0;
        end
        function kernel = uniform_circle(pixel_radius)
            arguments
                pixel_radius {mustBePositive,mustBeInteger}
            end
            kernel = coord.^2 + coord'.^2 < pixel_radius.^2;
        end
    end
    methods
        function obj = CyclicConv2Regularizer(kernel, slice_axis, condition_callback)
            arguments
                kernel {mustBeNonnegative}
                slice_axis {mustBeMember(slice_axis,'xyz')}
                condition_callback = @(~) true
            end
            obj.condition_callback = condition_callback;
            obj.kernel = kernel;
            obj.kernel_sum = sum(kernel,'all');
            obj.slice_axis = slice_axis - 'x' + 1;
        end
        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            grad = project(obj,grad);
        end
        function [arr,degree] = try_preprocess(obj, arr, iter_idx)
            [~,degree] = try_preprocess@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            arr = project(obj,arr);
        end
    end
    methods(Hidden)
        function arr = project(obj, arr)
            other_axis = rem([obj.slice_axis obj.slice_axis+1],3) + 1;
            if isempty(obj.arr_slice) || any(size(obj.arr_slice,other_axis) ~= size(arr,other_axis))
                vecdim = rem([obj.slice_axis obj.slice_axis + 1],3) + 1;
                obj.arr_slice = zeros(size(arr, vecdim));
            end
            for slice_num = 1:size(arr,obj.slice_axis)
                if obj.slice_axis == 1
                    obj.arr_slice(:) = reshape(arr(slice_num,:,:), size(obj.arr_slice));
                elseif obj.slice_axis == 2
                    obj.arr_slice(:) = reshape(arr(:,slice_num,:), size(obj.arr_slice));
                else
                    obj.arr_slice(:) = reshape(arr(:,:,slice_num), size(obj.arr_slice));
                end
                obj.arr_slice = cconv2(obj.arr_slice, obj.kernel)./obj.kernel_sum;
                if obj.slice_axis == 1
                    arr(slice_num,:,:) = obj.arr_slice;
                elseif obj.slice_axis == 2
                    arr(:,slice_num,:) = obj.arr_slice;
                else
                    arr(:,:,slice_num) = obj.arr_slice;
                end
            end
        end
    end
end
