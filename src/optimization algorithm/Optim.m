classdef Optim < handle
    methods
        function reset(~)
        end
        % apply gradient
        function arr_after = apply_gradient(~, arr_after, arr_before, gradient, step)
            arr_after(:) = arr_before + step * gradient;
        end
    end
end