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
        function [grad,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
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
            arr = mean(arr,obj.direction);
        end
    end
end
