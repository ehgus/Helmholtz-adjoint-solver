classdef BinaryRegularizer < Regularizer
    % See `Inverse Design of Nanophotonic Devices with Structural Integrity`
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
        function obj = BinaryRegularizer(max_val, min_val, unit_alpha, beta, condition_callback)
            arguments
                max_val
                min_val
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
        function A = conditional_apply(obj, A, iter_idx)
            degree = obj.condition_callback(iter_idx);
            if degree > 0
                A = apply(obj, A, degree);
            else
                A = apply_bound_only(obj, A);
            end
        end
        function A = apply_bound_only(obj, A)
            if isempty(obj.density_map) || all(size(obj.density_map) ~= size(A))
                obj.density_map = zeros(size(A));
            end
            obj.density_map(:) = real(A - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            A(:) = obj.density_map*(obj.max_val-obj.min_val) + obj.min_val;
        end
        function A = apply(obj, A, degree)
            if degree <= 0
                return
            end
            alpha = obj.unit_alpha * degree;
            tanh_val = tanh(alpha*obj.beta);
            % values into density map
            if isempty(obj.density_map) || all(size(obj.density_map) ~= size(A))
                obj.density_map = zeros(size(A));
            end
            obj.density_map(:) = real(A - obj.min_val)./real(obj.max_val-obj.min_val);
            obj.density_map(obj.density_map > 1) = 1;
            obj.density_map(obj.density_map < 0) = 0;
            % binarization
            obj.density_map = tanh_val + tanh(alpha.*(obj.density_map-obj.beta));
            obj.density_map = obj.density_map./(tanh_val + tanh(alpha*(1-obj.beta)));
            A(:) = obj.density_map*(obj.max_val-obj.min_val) + obj.min_val;
        end
    end
end