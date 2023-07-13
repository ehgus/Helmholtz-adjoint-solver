classdef Regularizer < handle
    properties
        condition_callback = @(step) true
    end
    methods(Abstract)
        A = apply(obj, A, degree)
    end
    methods
        function A = conditional_apply(obj, A, step)
            degree = obj.condition_callback(step);
            if degree ~= 0
                A = apply(obj, A, degree);
            end
        end
    end
end