classdef FieldGenerator < handle
    % generate array correponding to coherent plane light source in frequency space
    % Return array with size [Xsize, Ysize, 2 if vector_simulation else 1, illumination_number]
    properties (Constant)
        wavelength = NaN;           %wavelength
        NA = 1;                     %Numerical aperture
        percentage_NA_usage=0.95;   %Percent of Numerical aperture to use
        illumination_style='random';%illumination style: random, circle
        size = [NaN, NaN];          %XY size of illumination
        illumination_number=10;     %Number of illumination patterns to generate
        vector_simulation=true;     %Vectorial source or scalar wave source
        start_with_normal=true;     %Check whether you always put normal plane at the start of the illumination
    end
    methods (Static)
        function output_field = get_field(options)
            options = FieldGenerator.fill_default_parameters(options);
            pol_num = 1;
            if options.vector_simulation
                pol_num = 2;
            end
            output_field=zeros(options.size(1), options.size(2), pol_num, options.illumination_number, 'single');
            options.RI_bg = 1; % vacuum RI 
            utility = derive_optical_tool(options);
            for ill_num = 1:options.illumination_number
                d1 = 1;
                d2 = 1;
                switch options.illumination_style
                    case 'random'
                        while(sqrt(d1.^2+d2.^2)>1)
                            d1=2*(rand-1/2);
                            d2=2*(rand-1/2);
                        end
                    case 'circle'
                            dividend=1/(options.illumination_number);
                            if options.start_with_normal
                                dividend=1/(options.illumination_number-1);
                            end
                            d1=sin((ill_num-1)*2*pi*(dividend));
                            d2=cos((ill_num-1)*2*pi*(dividend));
                    otherwise
                        error("Unknown illumination style name")
                end
                %first angle should be normal
                if ill_num==1 && options.start_with_normal 
                    d1=0;
                    d2=0;
                end
                try
                    output_field(...
                        floor(size(output_field,1)/2)+1+round(d1.*options.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{1}),...
                        floor(size(output_field,2)/2)+1+round(d2.*options.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{2}),...
                        1,ill_num)=1;
                catch
                    error("An error occured while creating the field. can the resolution support the specified NA ?");
                end
            end
            output_field=fftshift(fft2(ifftshift(output_field)));
        end

        function options = fill_default_parameters(options)
            % fill default values if some mendatory values are missing
            instance_properties = setdiff(properties(FieldGenerator), fieldnames(options));
            for i = 1:length(instance_properties)
                property = instance_properties{i};
                options.(property) = FieldGenerator.(property);
            end
        end
    end
end