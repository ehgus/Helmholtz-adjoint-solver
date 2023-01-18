classdef RI_DB
    % Compact refractive index database
    properties
        db_location = fullfile(fileparts(mfilename('fullpath')),"RI_database");
    end
    methods
        function obj = RI_DB(new_db_location)
            if nargin == 1
                assert(isfolder(new_db_location), "The new database folder should exist")
                obj.db_location = new_db_location;
            end
        end
        function RI = get_RI(obj, material_name, wavelength)
            % return RI of the specific material in a given wavelength (um)
            % It reads table from database and return
            if numel(material_name) > 1
                RI = arrayfun(@(name)get_RI(obj,name, wavelength), material_name);
                return
            end
            RI = 1;
            if strcmp(material_name, "vacuum")
                return
            end
            RI_file = strcat(material_name, ".csv");
            RI_full_profile = readtable(fullfile(obj.db_location, RI_file));
            wl_list = RI_full_profile.wl;
            n_list = RI_full_profile.n;
            nan_loc = find(isnan(wl_list)); % imaginary part is specified by NaN value
            if isempty(nan_loc)
                n_fit = fit(wl_list, n_list, 'smoothingspline');
                RI = n_fit(wavelength);
            else
                wl_list = wl_list(1:nan_loc-1);
                k_list = n_list(nan_loc+1:end);
                n_list = n_list(1:nan_loc-1);
                n_fit = fit(wl_list, n_list, 'smoothingspline');
                k_fit = fit(wl_list, k_list,'smoothingspline');
                RI = n_fit(wavelength) + 1i*k_fit(wavelength);
            end
        end
    end
end