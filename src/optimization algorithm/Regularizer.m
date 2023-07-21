classdef (Abstract) Regularizer < handle
    properties
        condition_callback = @(step) true
    end
    methods(Abstract)
        A = preprocess(obj, A)
        A = postprocess(obj, A)
    end
    methods
        function [grad, arr] = regularize_gradient(obj, grad, arr, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree == 0
                return
            end
        end
        function A = regularize(obj, A, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree == 0
                return
            end
        end
    end
end