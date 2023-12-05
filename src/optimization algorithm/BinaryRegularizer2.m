classdef BinaryRegularizer2 < BoundRegularizer
    % Binarize RI by accelerating gradient depending on the current RI value
    %
    % acceleration = -[ρ log_2(ρ) + (1-ρ)log_2(1-ρ)]
    % where ρ = β'(density_map-1/2)+1/2 and β'= β^(-nα)
    % Then, grad = grad * acceleration
    properties
        unit_alpha
        beta
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
            obj@BoundRegularizer(min_val, max_val,condition_callback);
            obj.unit_alpha = unit_alpha;
            obj.beta = beta;
        end
        function [grad, degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [grad,degree] = regularize_gradient@BoundRegularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            reg_beta = min(obj.beta.^(1 - (degree-1)*obj.unit_alpha),1-10^-5);
            obj.density_map(:) = real(arr - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map = reg_beta.*(obj.density_map - 1/2)+1/2;
            obj.density_map = -obj.density_map.*log2(obj.density_map) - (1-obj.density_map).*log2(1-obj.density_map);
            obj.density_map = abs(obj.density_map); % in the case of digit error
            grad =grad.*obj.density_map.*(obj.max_val-obj.min_val);
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