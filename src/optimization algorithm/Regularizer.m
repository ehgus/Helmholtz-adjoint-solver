classdef (Abstract) Regularizer < handle
    % Regularizer constrains the values toward fabricable patterns
    % If it is executed at least once, `try_preprocess` and `try_postprocess` must be 
    % executed at the first and last only once each.
    properties
        condition_callback = @(step) true
    end
    properties(Hidden)
        activate_postprocess = false
    end
    methods
        function init(obj)
            obj.activate_postprocess = false;
        end
        function [arr,degree] = try_preprocess(obj,arr,iter_idx)
            degree = obj.condition_callback(iter_idx);
            if obj.activate_postprocess || degree <= 0
                degree = 0;
                return
            end
            obj.activate_postprocess = true;
        end
        function [arr,degree] = try_postprocess(obj, arr)
            degree = double(obj.activate_postprocess);
        end
        function [arr,degree] = interpolate(obj,arr,iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree <= 0
                return
            end
        end
        function [grad,arr,degree] = regularize_gradient(obj,grad,arr,iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree <= 0
                return
            end
        end
    end
end