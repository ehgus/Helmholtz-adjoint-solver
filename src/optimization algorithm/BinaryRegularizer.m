classdef BinaryRegularizer < Regularizer
    % Binarize RI by minimizing addional cost function
    % 
    % cost function = nα(ρ log_2(ρ) + (1-ρ)log_2(1-ρ))
    % where ρ = β(density_map-1/2)+1/2
    properties
        min_val
        max_val
        eta
        unit_beta
    end
    properties(Hidden)
        density_map
    end
    methods
        function obj = BinaryRegularizer(min_val, max_val, eta, unit_beta, condition_callback)
            arguments
                min_val
                max_val
                eta {mustBePositive} = 0.5
                unit_beta {mustBePositive} = 0.5
                condition_callback = @(~) true
            end
            obj.min_val = min_val;
            obj.max_val = max_val;
            obj.condition_callback = condition_callback; 
            obj.eta = eta;
            obj.unit_beta = unit_beta;
        end
        function [arr,degree] = regularize(obj,arr,iter_idx)
            [~,degree] = regularize@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            beta = degree.*obj.unit_beta;
            init_density_map(obj, arr);
            obj.density_map(:) = real(arr);
            denorminator = tanh(beta*obj.eta)+tanh(beta*(1-obj.eta));
            obj.density_map = tanh(beta.*(obj.density_map-obj.eta));
            obj.density_map = (tanh(beta*obj.eta)+obj.density_map)./denorminator;
            arr(:) = obj.density_map;
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
        function init_density_map(obj, arr)
            if isempty(obj.density_map) || any(size(obj.density_map,1:3) ~= size(arr,1:3))
                obj.density_map = zeros(size(arr));
            end
        end
    end
end