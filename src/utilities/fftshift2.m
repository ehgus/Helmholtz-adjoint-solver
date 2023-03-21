function mat = fftshift2(mat)
    mat = fftshift(fftshift(mat,1),2);
end