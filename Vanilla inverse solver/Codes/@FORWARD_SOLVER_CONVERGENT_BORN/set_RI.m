        function set_RI(h,RI)
            if isstruct(RI)
                RI=RI.scatt_pott;
                if size(RI,4)<=1
                    error('Structur form only allowed for tensor RI');
                end
            end
            sz_RI=size(RI);
            if ~isequal(sz_RI(1:3)',h.expected_RI_size(:))
                error(['The refractiv index does not have the expected size : ' ...
                    num2str(h.expected_RI_size(1)) ' ' num2str(h.expected_RI_size(2)) ' ' num2str(h.expected_RI_size(3))]);
            end
            RI=single(RI);%single computation are faster
            
            set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end


