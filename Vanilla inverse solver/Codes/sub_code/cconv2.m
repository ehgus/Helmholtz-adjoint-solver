function c = cconv2(a,kernel)
    %   CCONV modulo-N circular convolution
    %   C = CCONVN(A, B, N_size) circularly convolvs multidimensional arrays A and B.
    %   The resulting array has the same size with 'a'.
    %   The center of kernel should set to be fix(size_k/2)+1.
    size_a = size(a);
    size_k = size(kernel);
    assert(length(size_k) ==2, "The kernel should be a 2D array");
    padded_kernel = circshift(padarray(kernel,size_a(1:2) - size_k,'post'),-fix((size_k-1)/2));

    isReal = isreal(a)&&isreal(kernel);
    
    x = ifft2(fft2(a).*fft2(padded_kernel));
    
    if isReal
        c = real(x);
    else
        c = x;
    end
end