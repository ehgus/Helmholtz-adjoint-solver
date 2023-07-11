function A=fft_flip(A,flip_bool)
    % flip the array along the center of the array
    % The zero frequency is assumed to be placed at A(1,1,1,....)
    for idx = 1:length(flip_bool)
        if flip_bool(idx)
            A=flip(A,idx);
            A=circshift(A,1,idx);
        end
    end
end
