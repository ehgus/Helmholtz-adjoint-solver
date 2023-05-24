classdef Optim < handle
    methods
        function reset(obj)
        end
        % apply gradient
        function arr_after = apply_gradient(obj, arr_after, arr_before, gradient, step)
            arr_after(:) = arr_before + step * gradient;
        end
    end
end