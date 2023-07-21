classdef AvgRegularizer < Regularizer
    properties
        direction
    end
    methods
        function obj = AvgRegularizer(direction, condition_callback)
            arguments
                direction {mustBeMember(direction,'xyz')}
                condition_callback = @(~) true
            end
            obj.condition_callback = condition_callback;
            obj.direction = direction - 'x' + 1; % 'x' => 1, 'y' => 2, 'z' => 3
        end
        function avg_A = preprocess(obj, A)
            avg_A = mean(A, obj.direction);
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
            arr = preprocess(obj, arr);
        end
        function avg_A = regularize(obj, A, ~)
            avg_A = preprocess(obj, A);
        end
    end
end
