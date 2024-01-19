classdef MinimumLengthRegularizer < Regularizer
    % see "Minimum length scale in topology optimization by geometric constraints"
    properties
        % RI bound used to convert RImap into density map
        min_RI
        max_val
        %
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
        function obj = MinimumLengthRegularizer(min_RI, max_RI,weight, decay_rate,prob_threshold, condition_callback)
            arguments
                min_RI
                max_RI
                weight
                decay_rate {mustBePositive}
                prob_threshold
                condition_callback = @(~) true
            end
            obj.min_RI = min_RI;
            obj.max_val = max_RI;
            obj.condition_callback = condition_callback;
            obj.weight = weight;
            obj.decay_rate = decay_rate;
            obj.prob_threshold = prob_threshold;
        end
        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx);
            if degree <= 0
                return
            end
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(arr,1:3))
                obj.density_map = zeros(size(arr));
                obj.hessian_map = zeros(size(arr));
                obj.solid_inflection = zeros(size(arr));
                obj.void_inflection = zeros(size(arr));
            end
            obj.density_map(:) = real(arr);
            obj.hessian_map(:) = abs(cconv2(obj.density_map, [-1 0 1])).^2;
            obj.hessian_map = obj.hessian_map +abs(cconv2(obj.density_map, [-1 0 1]')).^2;
            obj.hessian_map = exp(-obj.decay_rate.*obj.hessian_map);
            obj.solid_inflection(:) = obj.density_map.*obj.hessian_map;
            obj.void_inflection(:) = (1-obj.density_map).*obj.hessian_map;
            % add gradient
            grad = grad - 2*obj.weight.*obj.solid_inflection.*min(obj.density_map-obj.prob_threshold(2),0);
            grad = grad + 2*obj.weight.*obj.void_inflection.*min(obj.prob_threshold(1)-obj.density_map,0);
        end
    end

end