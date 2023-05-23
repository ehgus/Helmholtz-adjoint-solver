classdef FistaOptim < Optim
    properties
        t_n = 0
        t_np = 1
        s_n = []
        x_n = []
    end
    methods
        function obj = reset(obj)
            obj.t_n = 0;
            obj.t_np = 1;
            obj.s_n = [];
            obj.x_n = [];
        end
        function arr_after = apply_gradient(obj, arr_after, arr_before, gradient, step)
            if isempty(obj.s_n)
                obj.s_n = arr_before;
                obj.x_n = arr_before;
            end
            obj.t_n = obj.t_np;
            obj.t_np = (1+sqrt(1+4*obj.t_n^2))/2;
            obj.s_n = apply_gradient@Optim(obj, obj.s_n, arr_before, gradient, step);
            arr_after(:) = obj.s_n + (obj.t_n - 1)/obj.t_np*(obj.s_n - obj.x_n);
            obj.x_n(:) = obj.s_n;
        end
    end
end