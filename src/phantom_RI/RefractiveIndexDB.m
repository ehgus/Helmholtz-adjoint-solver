classdef RefractiveIndexDB
    % Refractive index database
    % It provides the same refractive index profiles as https://refractiveindex.info/
    properties
        db_location = fullfile(fileparts(mfilename('fullpath')),"refractiveindex_database/database/data");
    end
    methods
        function obj = RefractiveIndexDB(new_db_location)
            if nargin == 1
                assert(isfolder(new_db_location), "The database folder should be in accessible")
                obj.db_location = new_db_location;
            end
        end
        function RI_func = material(obj, shelf, book, page)
            % provide RI profile given information
            % To learn how to know the input parameters, visit https://refractiveindex.info/
            % shelf: type of the material
            % book: name of the material
            % page: reference of the material's RI profile
            RI_profile = sprintf("%s/%s/%s",shelf,book,page);
            RI_info_path = fullfile(obj.db_location, strcat(RI_profile, ".yml"));
            assert(isfile(RI_info_path), sprintf("RI profile with %s does not exist", RI_profile))
            RI_info = yaml.loadFile(RI_info_path).DATA;
            RI_func_list = cell(1, length(RI_info));
            for idx = 1:length(RI_info)
                if strncmp(RI_info{idx}.type,'formula',7)
                    coeff = str2num(RI_info{idx}.coefficients);
                else % tabulated data
                    coeff = str2num(RI_info{idx}.data);
                end
                func_name = strcat("RefractiveIndexDB.",strrep(strip(RI_info{idx}.type),' ','_'));
                RI_func_list{idx} = feval(func_name, coeff);
            end
            RI_func = @(wavelength) sum(cellfun(@(func) func(wavelength), RI_func_list));
        end
    end

    methods (Static, Hidden)
        function RI_func = formula_1(c)
            % Sellmeier
            function RI = RI_func_nested(wavelength)
                RI_sqr = 1 + c(1);
                for idx = 2:2:length(c)
                    RI_sqr = RI_sqr + c(idx)./(1-(c(idx+1)./wavelength).^2);
                end
                RI = sqrt(RI_sqr);
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_2(c)
            % Sellmeier2
            function RI = RI_func_nested(wavelength)
                RI_sqr = 1 + c(1);
                for idx = 2:2:length(c)
                    RI_sqr = RI_sqr + c(idx)./(1-c(idx+1)./wavelength.^2);
                end
                RI = sqrt(RI_sqr);
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_3(c)
            % Polynomial
            function RI = RI_func_nested(wavelength)
                RI_sqr = c(1);
                for idx = 2:2:length(c)
                    RI_sqr = RI_sqr + c(idx).*wavelength.^c(idx+1);
                end
                RI = sqrt(RI_sqr);
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_4(c)
            % RIInfo
            function RI = RI_func_nested(wavelength)
                RI_sqr = c(1);
                for idx = 2:4:min(length(c), 9)
                    RI_sqr = RI_sqr + c(idx).*wavelength.^c(idx+1)./(wavelength.^2-c(idx+2)^c(idx+3));
                end
                for idx = 10:2:n
                    RI_sqr = RI_sqr + c(idx).*wavelength.^c(idx+1);
                end
                RI = sqrt(RI_sqr);
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_5(c)
            %Cauchy
            function RI = RI_func_nested(wavelength)
                RI = c(1);
                for idx = 2:2:length(c)
                    RI = RI + c(idx)*wavelength.^c(idx+1);
                end
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_6(c)
            %Gases
            function RI = RI_func_nested(wavelength)
                RI = 1 + c(1);
                for idx = 2:2:length(c)
                    RI = RI + c(idx)./(c(idx+1) - 1./wavelength.^2);
                end
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_7(c)
            %Herzberger
            function RI = RI_func_nested(wavelength)
                RI = c(1);
                RI = RI + c(2)./(wavelength.^2 - 0.028);
                RI = RI + c(3)./(wavelength.^2 - 0.028).^2;
                for idx = 4:length(c)
                    pwd = 2*(idx -3);
                    RI = RI + c(idx)*wavelength.^pwd;
                end
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_8(c)
            %Retro
            function RI = RI_func_nested(wavelength)
                sq_wl = wavelength.^2;
                RI = c(1) + (c(2)./(sq_wl - c(3)) + c(4)).*sq_wl;
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = formula_9(c)
            %Exotic
            function RI = RI_func_nested(wavelength)
                RI = c(1) + c(2)./(wavelength.^2 - c(3));
                RI = RI + c(4).*(wavelength - c(5))./((wavelength - c(5)).^2 + c(6));
            end
            RI_func = @RI_func_nested;
        end
        function RI_func = tabulated_n(RI_array)
            wl_list = RI_array(:,1);
            n_list = RI_array(:,2);
            RI_func = fit(wl_list, n_list, 'smoothingspline');
        end
        function RI_func = tabulated_k(RI_array)
            wl_list = RI_array(:,1);
            k_list = RI_array(:,2);
            k_fit = fit(wl_list, k_list, 'smoothingspline');
            RI_func = @(wavelength) 1i*k_fit(wavelength);
        end
        function RI_func = tabulated_nk(RI_array)
            n_func = RefractiveIndexDB.tabulated_n(RI_array(:,[1 2]));
            k_func = RefractiveIndexDB.tabulated_k(RI_array(:,[1 3]));
            RI_func = @(wavelength) n_func(wavelength) + k_func(wavelength);
        end
    end
end
