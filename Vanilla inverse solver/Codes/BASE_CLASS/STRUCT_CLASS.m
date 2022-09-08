classdef STRUCT_CLASS < handle
    % STRUCT_CLASS: take strcut to update properties of a instance
    properties
        verbose = false;
    end
    methods
        function h = update_properties(h, struct_parameters)
            warning off backtrace
            struct_names = fieldnames(struct_parameters);
            for i = 1:length(struct_names)
                name = struct_names{i}; % cell to char
                if ismember(name, properties(h))
                    h.(name) = struct_parameters.(name);
                else
                    warning(sprintf("%s does not have a property '%s'",class(h),name));
                end
            end
            warning on backtrace
        end
    end
end