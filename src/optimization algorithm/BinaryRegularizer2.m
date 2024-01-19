classdef BinaryRegularizer2 < Regularizer
    % Binarize RI by accelerating gradient depending on the current RI value
    %
    % acceleration = -[ρ log_2(ρ) + (1-ρ)log_2(1-ρ)]
    % where ρ = β'(density_map-1/2)+1/2 and β'= β^(-nα)
    % Then, grad = grad * acceleration
    properties
        min_val
        max_val
        unit_alpha
        beta
    end
    properties(Hidden)
        density_map
    end
    methods
        function obj = BinaryRegularizer2(min_val, max_val, unit_alpha, beta, condition_callback)
            arguments
                min_val
                max_val
                unit_alpha {mustBePositive} = 0.5
                beta {mustBeInRange(beta,0,1)} = 0.5
                condition_callback = @(~) true
            end
            obj.min_val = min_val;
            obj.max_val = max_val;
            obj.condition_callback = condition_callback; 
            obj.unit_alpha = unit_alpha;
            obj.beta = beta;
        end
        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            reg_beta = min(obj.beta.^(1 - (degree-1)*obj.unit_alpha),1-10^-5);
            obj.density_map(:) = real(arr);
            obj.density_map = reg_beta.*(obj.density_map - 1/2)+1/2;
            obj.density_map = -obj.density_map.*log2(obj.density_map) - (1-obj.density_map).*log2(1-obj.density_map);
            obj.density_map = abs(obj.density_map); % in the case of digit error
            grad =grad.*obj.density_map;
        end
        function [arr,degree] = try_postprocess(obj, arr)
            [~,degree] = try_postprocess@Regularizer(obj, arr);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            obj.density_map(:) = real(arr);
            obj.density_map(obj.density_map >= 0.5) = 1;
            obj.density_map(obj.density_map < 0.5) = 0;
            arr(:) = obj.density_map;
        end
    end
    methods(Hidden)
        function arr = bound_values(obj,arr)
            init_density_map(obj, arr);
            obj.density_map(:) = real((arr-obj.min_val)./(obj.max_val-obj.min_val));
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            arr(:) = obj.density_map;
        end
        function init_density_map(obj, arr)
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(arr,1:3))
                obj.density_map = zeros(size(arr));
            end
        end
    end
end