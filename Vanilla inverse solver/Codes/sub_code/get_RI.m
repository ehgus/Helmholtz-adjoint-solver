function value0 = get_RI(cd0,name0, wavelength)

    if strcmp(name0, "vacuum")
        value0 = 1;
    else
        RIfile = name0+ ".csv";
        M = readmatrix(fullfile(cd0, 'Codes', '@PHANTOM', 'Materials Complex RI', RIfile));
        wl = M(:,1); % [um]
        nan_loc = find(isnan(wl));
        if ~isempty(nan_loc)
            wl = wl(1:nan_loc-1);
            n = M(:,2); % 
            k = n(nan_loc+1:end);
            n = n(1:nan_loc-1);
            [f, ~, ~] = fit(wl,n,'smoothingspline');
            [g, ~, ~] = fit(wl,k,'smoothingspline');
            value0 = f(wavelength) + 1i*g(wavelength);
        else
            n = M(:,2); % 
            [f, ~, ~] = fit(wl,n,'smoothingspline');
            value0 = f(wavelength);
        end
    end
end