function RI = get_TiO2_mask(phantom_params)
%%
    outer_size = phantom_params.outer_size;
    cd0 = phantom_params.cd0;

%         d1=single(reshape(single(1:outer_size(1)),[],1,1)-(floor(outer_size(1)/2)+1));
%         d2=single(reshape(single(1:outer_size(2)),1,[],1)-(floor(outer_size(2)/2)+1));
        d3=single(reshape(single(1:outer_size(3)),1,1,[])-(floor(outer_size(3)/2)+1));
        RI = zeros(outer_size, 'single');

        thickness = round(phantom_params.thickness ./ phantom_params.resolution(3));
        wavelength = phantom_params.wavelength;
        for j1 = 1:length(phantom_params.name)
            if j1 == 1
                RI(:,:,1:thickness(j1)) = get_RI(cd0,phantom_params.name(j1), wavelength);
                end_idx = thickness(1);
            else
                %%
                RI(:,:,end_idx+1:end) = get_RI(cd0,phantom_params.name(j1), wavelength);
                end_idx = end_idx + thickness(j1);
            end
        end

end
