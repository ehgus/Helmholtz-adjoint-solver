function set_RI(h,RI)

    size_RI=size(RI);
    if ~isequal(size_RI(1:3)',h.expected_RI_size(:))
        error(['The refractiv index does not have the expected size : ' ...
            num2str(h.expected_RI_size(1)) ' ' num2str(h.expected_RI_size(2)) ' ' num2str(h.expected_RI_size(3))]);
    end
    RI=single(RI);%single computation are faster
    
    set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
    before_eps_imag = h.eps_imag;
    h.condition_RI();%modify the RI (add padding and boundary)
    after_eps_imag = h.eps_imag;
    if after_eps_imag < 10000*abs(before_eps_imag-after_eps_imag) % to prevent redundant Green function evaluation
        h.set_kernel();%init the parameter for the forward model
    end
end


