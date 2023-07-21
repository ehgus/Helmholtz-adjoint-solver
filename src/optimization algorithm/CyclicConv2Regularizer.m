classdef CyclicConv2Regularizer < Regularizer
    % See 'Inverse Design of Nanophotonic Devices with Structural Integrity'
    % or 'Photonic topology optimization with semiconductor-foundry design-rule constraints'
    properties
        kernel
        slice_axis
    end
    properties(Hidden)
        A_slice
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
        function A = preprocess(obj, A)
            other_axis = rem([obj.slice_axis obj.slice_axis+1],3) + 1;
            if isempty(obj.A_slice) || any(size(obj.A_slice,other_axis) ~= size(A,other_axis))
                vecdim = rem([obj.slice_axis obj.slice_axis + 1],3) + 1;
                obj.A_slice = zeros(size(A, vecdim));
            end
            for slice_num = 1:size(A,obj.slice_axis)
                if obj.slice_axis == 1
                    obj.A_slice(:) = reshape(A(slice_num,:,:), size(obj.A_slice));
                elseif obj.slice_axis == 2
                    obj.A_slice(:) = reshape(A(:,slice_num,:), size(obj.A_slice));
                else
                    obj.A_slice(:) = reshape(A(:,:,slice_num), size(obj.A_slice));
                end
                obj.A_slice = cconv2(obj.A_slice, obj.kernel)./obj.kernel_sum;
                if obj.slice_axis == 1
                    A(slice_num,:,:) = obj.A_slice;
                elseif obj.slice_axis == 2
                    A(:,slice_num,:) = obj.A_slice;
                else
                    A(:,:,slice_num) = obj.A_slice;
                end
            end
        end
        function A = postprocess(obj, A)
            A = preprocess(obj, A);
        end
        function [grad, arr]  = regularize_gradient(obj, grad, arr, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree == 0
                return
            end
            grad = preprocess(obj, grad);
        end
        function A = regularize(~, A, ~)
            return
        end
    end
end
