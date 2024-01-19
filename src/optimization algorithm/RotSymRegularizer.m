classdef RotSymRegularizer < Regularizer
    properties
        rot_axis
        angle_coeff
    end
    methods
        function obj = RotSymRegularizer(rot_axis,angle_coeff, condition_callback)
            arguments
                rot_axis (1,1) {mustBeMember(rot_axis,'xyz')}
                angle_coeff {mustBeInteger}
                condition_callback = @(~) true
            end
            obj.condition_callback = condition_callback;
            obj.rot_axis = rot_axis - 'x' + 1;
            obj.angle_coeff = angle_coeff;
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
            permute_dim = rem([obj.rot_axis obj.rot_axis+1 1],3) + 1;
            permute_dim(3) = obj.rot_axis;
            inverse_permute_dim = permute(1:3, permute_dim);
            permute_arr = permute(arr, permute_dim);
            for slice_idx = 1:size(permute_arr,3)
                for k = obj.angle_coeff
                    slice_arr = permute_arr(:,:,slice_idx);
                    permute_arr(:,:,slice_idx) = slice_arr + rot90(slice_arr, k);
                end
            end
            permute_arr = permute_arr./(2^length(obj.angle_coeff));
            arr = permute(permute_arr, inverse_permute_dim);
        end
    end
end