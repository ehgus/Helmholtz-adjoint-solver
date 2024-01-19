classdef Optim < handle
    properties
        % weight
        grad_weight {mustBePositive} = 0.1
        % regularization
        optim_region logical
        optim_ROI
        regularizer_sequence
    end
    methods
        function obj = Optim(optim_region, regularizer_sequence, grad_weight)
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
            if nargin > 2
                obj.grad_weight = grad_weight;
            end
            reset(obj);
        end

        function reset(obj)
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                reset(regularizer);
            end
        end

        function arr = try_preprocess(obj, arr,iter_idx)
            arr_region = arr(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            size_before = size(arr_region);
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                arr_region = try_preprocess(regularizer, arr_region, iter_idx);
            end
            size_after = size(arr_region,1:3);
            arr_region = repmat(arr_region, size_before./size_after);
            arr(obj.optim_region) = arr_region(obj.optim_region(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:));
        end
        
        function arr = try_postprocess(obj, arr)
            arr_region = arr(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            size_before = size(arr_region);
            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                arr_region = try_postprocess(regularizer, arr_region);
            end
            size_after = size(arr_region,1:3);
            arr_region = repmat(arr_region, size_before./size_after);
            arr(obj.optim_region) = arr_region(obj.optim_region(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:));
        end

        function arr = apply_gradient(obj, arr, grad, iter_idx)
            grad = obj.regularize_gradient(grad, arr, iter_idx);
            arr(obj.optim_region) = arr(obj.optim_region) - obj.grad_weight .* grad(obj.optim_region);
            arr = obj.regularize(arr, iter_idx);
        end
        
        function grad = regularize_gradient(obj, grad, arr, iter_idx)
            grad_region = grad(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            arr_region = arr(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            size_before = size(grad_region);

            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                [grad_region,arr_region] = regularize_gradient(regularizer, grad_region, arr_region, iter_idx);
            end
            size_after = size(grad_region,1:3);
            grad_region = repmat(grad_region, size_before./size_after);
            grad(~obj.optim_region) = 0;
            grad(obj.optim_region) = grad_region(obj.optim_region(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:));
        end

        function arr = regularize(obj, arr, iter_idx)
            arr_region = arr(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:);
            size_before = size(arr_region);

            for idx = 1:length(obj.regularizer_sequence)
                regularizer = obj.regularizer_sequence{idx};
                arr_region = regularize(regularizer, arr_region, iter_idx);
            end
            size_after = size(arr_region,1:3);
            arr_region = repmat(arr_region, size_before./size_after);
            arr(obj.optim_region) = arr_region(obj.optim_region(obj.optim_ROI(1,1):obj.optim_ROI(2,1),obj.optim_ROI(1,2):obj.optim_ROI(2,2),obj.optim_ROI(1,3):obj.optim_ROI(2,3),:,:));
        end
    end
end