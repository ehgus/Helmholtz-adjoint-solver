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
        function A = preprocess(obj, A)
            permute_dim = rem([obj.rot_axis obj.rot_axis+1 1],3) + 1;
            permute_dim(3) = obj.rot_axis;
            inverse_permute_dim = permute(1:3, permute_dim);
            
            permute_A = permute(A, permute_dim);
            for slice_idx = 1:size(permute_A,3)
                for k = obj.angle_coeff
                    slice_A = permute_A(:,:,slice_idx);
                    permute_A(:,:,slice_idx) = slice_A + rot90(slice_A, k);
                end
            end
            permute_A = permute_A./(2^length(obj.angle_coeff));
            A = permute(permute_A, inverse_permute_dim);
        end
        function A = postprocess(~, A)
            return
        end
        function [grad, arr] = regularize_gradient(obj, grad, arr, iter_idx)
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