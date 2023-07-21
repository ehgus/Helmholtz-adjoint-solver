classdef MinimumLengthRegularizer < Regularizer
    % see "Minimum length scale in topology optimization by geometric constraints"
    properties
        min_val
        max_val
        weight
        decay_rate
        prob_threshold
    end
    properties
        density_map
        hessian_map
        solid_inflection
        void_inflection
    end
    methods
        function obj = MinimumLengthRegularizer(min_val, max_val,weight, decay_rate,prob_threshold, condition_callback)
            arguments
                min_val
                max_val
                weight
                decay_rate {mustBePositive}
                prob_threshold
                condition_callback = @(~) true
            end
            obj.min_val = min_val;
            obj.max_val = max_val;
            obj.condition_callback = condition_callback;
            obj.weight = weight;
            obj.decay_rate = decay_rate;
            obj.prob_threshold = prob_threshold;
        end
        function A = preprocess(~, A)
        end
        function A = postprocess(obj, A)
        end
        function [grad, A] = regularize_gradient(obj, grad, A, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree == 0
                return
            end
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(A,1:3))
                obj.density_map = zeros(size(A));
                obj.hessian_map = zeros(size(A));
                obj.solid_inflection = zeros(size(A));
                obj.void_inflection = zeros(size(A));
            end
            obj.density_map(:) = real(A - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.hessian_map(:) = abs(cconv2(obj.density_map, [-1 0 1])).^2;
            obj.hessian_map = obj.hessian_map +abs(cconv2(obj.density_map, [-1 0 1]')).^2;
            obj.hessian_map = exp(-obj.decay_rate.*obj.hessian_map);
            obj.solid_inflection(:) = obj.density_map.*obj.hessian_map;
            obj.void_inflection(:) = (1-obj.density_map).*obj.hessian_map;
            % add gradient
            grad = grad - 2*obj.weight*(obj.max_val-obj.min_val).*obj.solid_inflection.*min(obj.density_map-obj.prob_threshold(2),0);
            grad = grad + 2*obj.weight*(obj.max_val-obj.min_val).*obj.void_inflection.*min(obj.prob_threshold(1)-obj.density_map,0);
        end
        function A = regularize(obj, A, iter_idx)
        end
    end

end