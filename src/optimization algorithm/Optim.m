classdef Optim < handle
    % < optimization process >
    %    
    %       grad -->--[regularize_grad]-->-- regularized grad
    %        ^                                      v            
    %        |                               [feedback grad]
    %        |                                      v
    %  [calculate grad]                        new density
    %        ^                                      v
    %        |                                 [projection]
    %        |                                      v
    %        RI --<--[interpolate density]--<-- density projected
    %
    properties
        % weight
        grad_weight {mustBePositive} = 0.1
        % regularization
        optim_region logical
        optim_ROI
        regularizer_sequence
        density_projection_sequence
        % intermediate representation
        density
    end
    methods
        function obj = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight)
            optim_ROI = zeros(2,3);
            for axis = 1:3
                vecdim = rem([axis, axis+1],3) + 1;
                nonzero_element = any(optim_region,vecdim);
                roi_start = find(nonzero_element(:),1,"first");
                roi_end = find(nonzero_element(:),1,"last");
                optim_ROI(:,axis) = [roi_start roi_end];
            end
            obj.optim_region = optim_region;
            obj.optim_ROI = optim_ROI;
            obj.regularizer_sequence = regularizer_sequence;
            obj.density_projection_sequence = density_projection_sequence;
            if nargin > 2
                obj.grad_weight = grad_weight;
            end
            init(obj);
        end

        function init(obj)
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                init(regularizer);
            end
            obj.density = [];
        end

        function RI = try_preprocess(obj, RI, iter_idx)
            % RI -> density
            if isempty(obj.density)
                obj.density = RI(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            end
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                obj.density = try_preprocess(regularizer, obj.density, iter_idx);
            end
            RI = interpolate(obj, RI, iter_idx);
        end

        function RI = try_postprocess(obj, RI)
            % density -> RI
            arr_part = obj.density;
            for idx = length(obj.regularizer_sequence):-1:1
                regularizer = obj.regularizer_sequence{idx};
                arr_part = try_postprocess(regularizer, arr_part);
            end
            RI(obj.optim_region) = arr_part;
        end

        function RI = interpolate(obj, RI, iter_idx)
            % density -> intermediate RI
            arr_part = obj.density;
            for idx = length(obj.regularizer_sequence):-1:1
                regularizer = obj.regularizer_sequence{idx};
                arr_part = interpolate(regularizer, arr_part, iter_idx);
            end
            RI(obj.optim_region) = arr_part;
        end

        function density_grad = regularize_gradient(obj, grad, RI, iter_idx)
            % grad w.r.t intermediate RI -> grad w.r.t density
            density_grad = grad(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            RI_region = RI(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            figure(3)%doritos
            subplot(2,length(obj.regularizer_sequence)+1,1);imagesc(real(density_grad(:,:,1)))
            subplot(2,length(obj.regularizer_sequence)+1,length(obj.regularizer_sequence)+2);imagesc(real(RI_region(:,:,1)))
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                [density_grad,RI_region] = regularize_gradient(regularizer, density_grad, RI_region, iter_idx);
                subplot(2,length(obj.regularizer_sequence)+1,idx+1);imagesc(real(density_grad(:,:,1)))
                subplot(2,length(obj.regularizer_sequence)+1,length(obj.regularizer_sequence)+idx+2);imagesc(real(RI_region(:,:,1)))
            end
        end

        function density = project_density(obj, density, iter_idx)
            % density -> density projected
            for idx = 1:length(obj.density_projection_sequence)
                regularizer = obj.density_projection_sequence{idx};
                density = interpolate(regularizer, density, iter_idx);
            end
        end

        function RI = apply_gradient(obj, RI, grad, iter_idx)
            % grad for intermediate RI -> grad for density 
            density_grad = obj.regularize_gradient(grad, RI, iter_idx);
            obj.density = obj.density - obj.grad_weight .* density_grad;
            obj.density = project_density(obj, obj.density, iter_idx);
            RI = obj.interpolate(RI, iter_idx);
        end
    end
end