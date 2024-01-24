classdef BoundRegularizer < Regularizer
    % give addtional weight using entropy
    properties
        max_val
        min_val
    end
    properties(Hidden)
        density_map
    end
    methods
        function obj = BoundRegularizer(min_val, max_val, condition_callback)
            arguments
                min_val
                max_val
                condition_callback = @(~) true
            end
            obj.min_val = min_val;
            obj.max_val = max_val;
            obj.condition_callback = condition_callback; 
        end

        function [arr,degree] = try_preprocess(obj,arr,iter_idx)
            [~,degree] = try_preprocess@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            arr = arr2density(obj, arr);
        end

        function [arr,degree] = try_postprocess(obj,arr)
            [~,degree] = try_postprocess@Regularizer(obj,arr);
            if degree <= 0
                return
            end
            arr = density2arr(obj, arr);
        end

        function [arr,degree] = interpolate(obj,arr,iter_idx)
            [~,degree] = interpolate@Regularizer(obj, arr, iter_idx);
            if degree <= 0
                return
            end
            arr = density2arr(obj, arr);
        end

        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            grad = real(grad./(obj.max_val-obj.min_val));
            arr = arr2density(obj,arr);
        end
    end
    methods(Hidden)
        function density = arr2density(obj,arr)
            init_density_map(obj, arr);
            obj.density_map(:) = real((arr-obj.min_val)./(obj.max_val-obj.min_val));
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            density = obj.density_map;
        end
        function arr = density2arr(obj,density)
            init_density_map(obj, density);
            obj.density_map(:) = real(density);
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            arr = obj.density_map*(obj.max_val-obj.min_val)+obj.min_val;
        end
        function init_density_map(obj, arr)
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(arr,1:3))
                obj.density_map = zeros(size(arr));
            end
        end
    end
end