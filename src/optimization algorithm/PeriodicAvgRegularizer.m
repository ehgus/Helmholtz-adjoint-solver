classdef PeriodicAvgRegularizer < AvgRegularizer
    properties
        pixel_period
    end
    methods
        function obj = PeriodicAvgRegularizer(direction, pixel_period, condition_callback)
            arguments
                direction {mustBeMember(direction,'xyz')}
                pixel_period {mustBeInteger, mustBePositive}
                condition_callback = @(~) true
            end
            obj@AvgRegularizer(direction, condition_callback);
            obj.pixel_period = pixel_period;
        end
    end
    methods(Hidden)

        function arr = project(obj, arr)
            % take mean & apply
            period = obj.pixel_period;
            for i = 1:floor(size(arr,obj.direction)/period)
                if obj.direction == 1 % x direction
                    arr_tmp = mean(arr((period*(i-1)+1):(period*i),:),obj.direction);
                    for idx = (period*(i-1)+1):(period*i)
                        arr(idx,:) = arr_tmp;
                    end
                elseif obj.direction == 2 % y direction
                    arr_tmp = mena(arr(:,(period*(i-1)+1):(period*i),:),obj.direction);
                    for idx = (period*(i-1)+1):(period*i)
                        arr(:,idx,:) = arr_tmp;
                    end
                else % z direction
                    arr_tmp = mean(arr(:,:,(period*(i-1)+1):(period*i),:),obj.direction);
                    for idx = (period*(i-1)+1):(period*i)
                        arr(:,:,idx,:) = arr_tmp;
                    end
                end
            end
            % take mean & apply for remains
            if period*floor(size(arr,obj.direction)/period) ~= size(arr,obj.direction)
                warning("Current array size is not divisible by the proposed period")
                if obj.direction == 1 % x direction
                    arr_tmp = mean(arr((period*floor(size(arr,obj.direction)/period)+1):end,:),obj.direction);
                    for idx = (period*floor(size(arr,obj.direction)/period)+1):size(arr,obj.direction)
                        arr(idx,:) = arr_tmp;
                    end
                elseif obj.direction == 2 % y direction
                    arr_tmp = mena(arr(:,(period*floor(size(arr,obj.direction)/period)+1):end,:),obj.direction);
                    for idx = (period*floor(size(arr,obj.direction)/period)+1):size(arr,obj.direction)
                        arr(:,idx,:) = arr_tmp;
                    end
                else % z direction
                    arr_tmp = mean(arr(:,:,(period*floor(size(arr,obj.direction)/period)+1):end,:),obj.direction);
                    for idx = (period*floor(size(arr,obj.direction)/period)+1):size(arr,obj.direction)
                        arr(:,:,idx,:) = arr_tmp;
                    end
                end
            end
        end
    end
end
