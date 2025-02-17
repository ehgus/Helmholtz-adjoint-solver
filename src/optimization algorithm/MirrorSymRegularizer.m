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

        function [arr,degree] = try_preprocess(obj, arr, iter_idx)
            [~,degree] = try_preprocess@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            arr = project(obj,arr);
        end

        function [arr,degree] = try_postprocess(obj, arr)
            [~,degree] = try_postprocess@Regularizer(obj,arr);
            if degree <= 0
                return
            end
            arr = project(obj,arr);
        end

        function [arr,degree] = interpolate(obj, arr, iter_idx)
            [~,degree] = interpolate@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            arr = project(obj,arr);
        end

        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            grad = project(obj,grad);
        end
    end
    methods(Hidden)
        function arr = project(obj, arr)
            for axis = obj.direction
                arr = arr + flip(arr, axis);
            end
            arr = arr./(2^length(obj.direction));
        end
    end
end