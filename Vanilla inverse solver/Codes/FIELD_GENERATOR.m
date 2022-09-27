classdef FIELD_GENERATOR < OPTICAL_SIMULATION
    methods
        function get_default_parameters(h)
            get_default_parameters@OPTICAL_SIMULATION(h);
            %SIMULATION PARAMETERS
            h.parameters.percentage_NA_usage=0.95;
            h.parameters.illumination_style='random';% can be random circular etc...
            h.parameters.illumination_number=10;
            h.parameters.illumination_pol=[];
            h.parameters.start_with_normal=true;
        end
        function h=FIELD_GENERATOR(params)
            h@OPTICAL_SIMULATION(params);
        end
        function output_field=get_fields(h)
            % generate coherent light source
            
            if h.parameters.vector_simulation
                output_field=zeros(h.parameters.size(1),h.parameters.size(2),2,h.parameters.illumination_number,'single');
            else
                output_field=zeros(h.parameters.size(1),h.parameters.size(2),1,h.parameters.illumination_number,'single');
            end
            warning('off','all');
            utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            for ill_num=1:h.parameters.illumination_number
                d1=1;
                d2=1;
                
                switch h.parameters.illumination_style
                    case 'random'
                        while(sqrt(d1.^2+d2.^2)>1)
                            d1=2*(rand-1/2);
                            d2=2*(rand-1/2);
                        end
                    case 'circle'
                            dividend=1/(h.parameters.illumination_number);
                            if h.parameters.start_with_normal
                                dividend=1/(h.parameters.illumination_number-1);
                            end
                            d1=sin((ill_num-1)*2*pi*(dividend));
                            d2=cos((ill_num-1)*2*pi*(dividend));
                    otherwise
                        error("Unknown illumination style name")
                end
                %first angle should be normal
                if ill_num==1 && h.parameters.start_with_normal 
                    d1=0;
                    d2=0;
                end
                try
                    output_field(...
                        floor(size(output_field,1)/2)+1+round(d1.*h.parameters.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{1}),...
                        floor(size(output_field,2)/2)+1+round(d2.*h.parameters.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{2}),...
                        1,ill_num)=1;
                catch
                    error("An error occured while creating the field. can the resolution support the specified NA ?");
                end
            end
            output_field=fftshift(fft2(ifftshift(output_field)));
            
        end
    end
end