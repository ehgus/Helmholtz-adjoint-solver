classdef FistaOptim < Optim
    properties
        t_prev
        t_curr
        x_prev
        x_curr
    end
    methods
        function obj = FistaOptim(optim_region, regularizer_sequence, grad_weight)
            if nargin < 3
                grad_weight = 0.1;
            end
            obj@Optim(optim_region, regularizer_sequence, grad_weight);
        end
        function obj = reset(obj)
            reset@Optim(obj);
            obj.t_prev = 0;
            obj.t_curr = 1;
            obj.x_curr = [];
            obj.x_prev = [];
        end
        function arr = apply_gradient(obj, arr, gradient, iter_idx)
            if isempty(obj.x_curr)
                obj.x_prev = arr;
                obj.x_curr = arr;
            end
            % x_k = p_L(y_k)
            obj.x_curr(:) = arr;
            obj.x_curr(obj.optim_region) = arr(obj.optim_region) + obj.grad_weight .* gradient(obj.optim_region);
            % t_k = (1 + sqrt(1+4t_{k-1}^2)/2)
            obj.t_prev = obj.t_curr;
            obj.t_curr = (1+sqrt(1+4*obj.t_prev^2))/2;
            % y_{k+1} = x_{k} + (t_{k-1} - 1)/t_k*(x_k - x_{k-1}) = (1 + (t_{k-1} - 1)/t_k)*x_k - (t_{k-1} - 1)/t_k*x_{k-1}
            fista_weight = (obj.t_prev - 1)/obj.t_curr;
            arr(:) = (1 + fista_weight).*obj.x_curr;
            arr = arr - fista_weight.*obj.x_prev;
            arr = obj.regularize_pattern(arr, iter_idx);
            obj.x_prev(:) = obj.x_curr;
        end
    end
end