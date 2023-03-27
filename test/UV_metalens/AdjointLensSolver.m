classdef AdjointLensSolver < AdjointSolver
    properties
        z_bound_pixel
    end
    methods
        function obj = AdjointLensSolver(options)
            obj@AdjointSolver(options);
            obj.z_bound_pixel = any(obj.ROI_change,[1 2]);
        end
        function RI_opt = post_regularization(obj,RI_opt,index)
            RI_opt = post_regularization@AdjointSolver(obj,RI_opt,index);
            % circular averaging
            RI_sub = RI_opt(:,:,obj.z_bound_pixel);
            RI_sub = RI_sub + flip(RI_sub,1);
            RI_sub = RI_sub + flip(RI_sub,2);
            for i = 1:size(RI_sub,3)
                RI_sub(:,:,i) = RI_sub(:,:,i) + transpose(RI_sub(:,:,i));
            end
            RI_opt(:,:,obj.z_bound_pixel) = RI_sub/8;
        end
    end
end