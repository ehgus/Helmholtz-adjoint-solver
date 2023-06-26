function array=fft_flip(array,flip_bool, use_shift)
%use_shift is set to true if the array has the zero frequency centered
for idx = 1:length(flip_bool)
    if flip_bool(idx)
        array=flip(array,idx);
        if mod(size(array,idx),2)==0 || ~use_shift
            array=circshift(array,1,idx);
        end
    end
end
end
