classdef BinaryRegularizer2 < Regularizer
    % give addtional weight using entropy
    properties
        unit_alpha
        beta
        max_val
        min_val
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
            obj.condition_callback = condition_callback; 
            obj.max_val = max_val;
            obj.min_val = min_val;
            obj.unit_alpha = unit_alpha;
            obj.beta = beta;
        end
        function A = preprocess(obj, A)
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(A,1:3))
                obj.density_map = zeros(size(A));
            end
            obj.density_map(:) = real(A - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            A(:) = obj.density_map*(obj.max_val-obj.min_val) + obj.min_val;
        end
        function A = postprocess(obj, A)
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(A,1:3))
                obj.density_map = zeros(size(A));
            end
            obj.density_map(:) = real(A - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map(obj.density_map >= 0.5) = 1;
            obj.density_map(obj.density_map < 0.5) = 0;
            A(:) = obj.density_map*(obj.max_val-obj.min_val) + obj.min_val;
        end
        function [grad, arr] = regularize_gradient(obj, grad, arr, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree > 0
                reg_beta = min(obj.beta.^(1 - (degree-1)*obj.unit_alpha),1-10^-5);
                obj.density_map(:) = real(arr - obj.min_val)./real(obj.max_val-obj.min_val);
                obj.density_map = reg_beta.*(obj.density_map - 1/2)+1/2;
                obj.density_map = -obj.density_map.*log2(obj.density_map) - (1-obj.density_map).*log2(1-obj.density_map);
                obj.density_map = abs(obj.density_map); % in the case of digit error
                obj.density_map = obj.density_map.*real(grad)./real(obj.max_val-obj.min_val);
            else
                obj.density_map(:) = real(grad)./real(obj.max_val-obj.min_val);
            end
            grad(:) = obj.density_map.*(obj.max_val-obj.min_val);
        end
        function A = regularize(obj, A, ~)
            A = preprocess(obj, A);
        end
    end
end