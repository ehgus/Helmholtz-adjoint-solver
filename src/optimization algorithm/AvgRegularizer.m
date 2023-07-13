classdef AvgRegularizer < Regularizer
    properties
        direction
    end
    methods
        function obj = AvgRegularizer(direction, condition_callback)
            arguments
                direction {mustBeMember(direction,'xyz')}
                condition_callback = @(~) true
            end
            obj.condition_callback = condition_callback;
            obj.direction = direction - 'x' + 1; % 'x' => 1, 'y' => 2, 'z' => 3
        end
        function avg_A = apply(obj, A, ~)
            avg_A = mean(A, obj.direction);
        end
    end
end
