classdef ADJOINT_SOLVER < STRUCT_CLASS
    properties %(SetAccess = protected, Hidden = true)
        forward_solver;
        mode = 'Intensity';
    end
    methods
        Field=solve_adjoint(h,incident_field, intensity_mask)

        function h=ADJOINT_SOLVER(params)
            h = update_properties(h, params);
            if isempty(h.forward_solver)
                error('Forward solver should be specified');
            end
        end

        function set_RI(h,RI)
            h.forward_solver.set_RI(RI);
        end
    end
end