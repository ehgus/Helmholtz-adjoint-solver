classdef BinaryRegularizer < Regularizer
    % Binarize RI by minimizing addional cost function
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

        function [arr,degree] = interpolate(obj,arr,iter_idx)
            [~,degree] = interpolate@Regularizer(obj,arr,iter_idx);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            beta = degree.*obj.unit_beta;
            obj.density_map(:) = real(arr);

            denorminator = tanh(beta*obj.eta)+tanh(beta*(1-obj.eta));
            obj.density_map = tanh(beta.*(obj.density_map-obj.eta));
            obj.density_map = (tanh(beta*obj.eta)+obj.density_map)./denorminator;
            arr(:) = obj.density_map;
        end

        function [grad,arr,degree] = regularize_gradient(obj, grad, arr, iter_idx)
            [~,~,degree] = regularize_gradient@Regularizer(obj, grad, arr, iter_idx-1);
            if degree <= 0
                return
            end
            init_density_map(obj, arr);
            obj.density_map(:) = real(arr);
            beta = degree.*obj.unit_beta;

            denorminator = tanh(beta*obj.eta)+tanh(beta*(1-obj.eta));
            obj.density_map = obj.density_map.*denorminator;
            obj.density_map = obj.density_map-tanh(beta*obj.eta);
            obj.density_map = atanh(obj.density_map)./beta+obj.eta;
            arr = obj.density_map;

            obj.density_map = beta.*sech(obj.density_map).^2;
            grad = grad.*obj.density_map./denorminator;
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