classdef BinaryRegularizer < BoundRegularizer
    % Binarize RI by minimizing addional cost function
    % 
    % cost function = nα(ρ log_2(ρ) + (1-ρ)log_2(1-ρ))
    % where ρ = β(density_map-1/2)+1/2
    properties
        unit_alpha
        beta
    end
    methods
        function obj = BinaryRegularizer(min_val, max_val, unit_alpha, beta, condition_callback)
            arguments
                min_val
                max_val
                unit_alpha {mustBePositive} = 0.5
                beta {mustBeInRange(beta,0,1)} = 0.5
                condition_callback = @(~) true
            end
            obj@BoundRegularizer(min_val, max_val,condition_callback);
            obj.unit_alpha = unit_alpha;
            obj.beta = beta;
        end
        function [grad,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [grad,degree] = regularize_gradient@BoundRegularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            obj.density_map(:) = real(arr - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map = obj.beta.*(obj.density_map - 1/2) + 1/2;
            obj.density_map = obj.beta.*log2((obj.density_map)./(1-obj.density_map));
            obj.density_map = (degree .* obj.unit_alpha) .* obj.density_map;
            grad = grad + obj.density_map.*(obj.max_val-obj.min_val);
        end
        function [arr,degree] = try_postprocess(obj, arr)
            [~,degree] = try_postprocess@BoundRegularizer(obj, arr);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            obj.density_map(:) = real(arr - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map(obj.density_map >= 0.5) = 1;
            obj.density_map(obj.density_map < 0.5) = 0;
            arr(:) = obj.density_map*(obj.max_val-obj.min_val) + obj.min_val;
        end
    end
end