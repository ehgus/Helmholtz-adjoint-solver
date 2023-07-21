classdef MirrorSymRegularizer < Regularizer
    properties
        direction
    end
    methods
        function obj = MirrorSymRegularizer(direction, condition_callback)
            arguments
                direction {mustBeMember(direction,'xyz')}
                condition_callback = @(~) true
            end
            obj.condition_callback = condition_callback;
            obj.direction = direction - 'x' + 1;
        end
        function A = preprocess(obj, A)
            for axis = obj.direction
                A = A + flip(A, axis);
            end
            A = A./(2^length(obj.direction));
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