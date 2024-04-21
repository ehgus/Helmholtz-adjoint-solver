function set_RI(obj, RI)

    size_RI=size(RI);
    if ~isequal(size_RI(1:3)',obj.expected_RI_size(:))
        error(['The refractive index does not have the expected size : ' ...
            num2str(obj.expected_RI_size(1)) ' ' num2str(obj.expected_RI_size(2)) ' ' num2str(obj.expected_RI_size(3))]);
    end
    obj.RI=single(RI);%single computation are faster
    obj.condition_RI();%modify the RI (add padding and boundary)
    obj.set_kernel();%init the parameter for the forward model
end


