clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% User-defined optical parameter
use_GPU = true; % accelerator option
NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.025;  % unit: micron
diameter = 10;       % unit: micron
focal_length = 3; % unit: micron
z_padding = 3;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'

% Refractive index profile
RI_database = RefractiveIndexDB();
substrate = RI_database.material("other","resists","Microchem SU-8 2000");
plate_thickness = 0.15;
PDMS = RI_database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = RI_database.material("main","TiO2","Siefke");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 substrate});
thickness_pixel = round([wavelength plate_thickness (focal_length + z_padding)]/resolution);
diameter_pixel = ceil(diameter/resolution);
RImap = phantom_plate([diameter_pixel diameter_pixel sum(thickness_pixel)], RI_list, thickness_pixel);

% Optical parameters
params.NA=NA;                   % Numerical aperture
params.wavelength=wavelength;   % unit: micron
[minRI, maxRI] = bounds(RI_list);
params.RI_bg=minRI;            % Background RI
params.resolution=ones(1,3) * resolution;         % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.size=size(RImap);        % 3D volume grid

% Incident field
source_params = params;
source_params.polarization = [1 -1i 0]/sqrt(2);
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;

current_source = PlaneSource(source_params);

%% Forward solver
params_CBS=params;
params_CBS.use_GPU=use_GPU;
params_CBS.boundary_thickness = [0 0 3];
params_CBS.field_attenuation = [0 0 3];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

forward_solver=ConvergentBornSolver(params_CBS);

% Configuration for bulk material
display_RI_Efield(forward_solver,RImap,current_source,'before optimization')

%% Adjoint method

% Target design paramters
radius_pixel = round(0.1/resolution);
one_turn_length = round(2.5/resolution);
distance = round(0.65/resolution);
num_helix = 2;
intensity_weight = phantom_multi_helix([diameter_pixel, diameter_pixel, round(2*z_padding/resolution)], [0 1], radius_pixel, one_turn_length, distance, num_helix);
intensity_weight = intensity_weight.*reshape(linspace(1,0.35,size(intensity_weight,3)),1,1,[]);
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
filter_axis = exp(-(-2:2).^2);
blur_filter = reshape(filter_axis,[],1).*reshape(filter_axis,1,[]).*reshape(filter_axis,1,1,[]);
blur_filter = blur_filter./sum(blur_filter,'all');
intensity_weight = convn(intensity_weight, blur_filter, 'same');
options.intensity_weight = intensity_weight;

filter = CyclicConv2Regularizer.conic(round(0.2/resolution));
optim_region = real(RImap) > 2;

%% Design case: single plate model
regularizer_sequence = { ...
    BoundRegularizer(RI_list(1), RI_list(2)), ...
    BinaryRegularizer(RI_list(1), RI_list(2), 0.5, 1, @(step) (step>=50)*(1+(step-50)/20)), ...
    CyclicConv2Regularizer(filter,'z', @(step) 15<=step) ...
};
density_projection_sequence = { ...
    BoundRegularizer(0, 1), ...
    AvgRegularizer('z', @(step) 15<=step), ...
    RotSymRegularizer('z',2, @(step) 15<=step), ...
};
grad_weight = 60;

adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 70;
adjoint_params.optimizer = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight);
adjoint_params.verbose = true;
adjoint_params.verbose_level = 0;
adjoint_params.sectioning_axis = "z";
adjoint_params.sectioning_position = thickness_pixel(1)+1;
adjoint_params.temp_save_dir = "temp_double_helix_single_plate";
adjoint_params.verbose_level = 1;
adjoint_params.temp_save_dir = "temp_double_helix";

adjoint_solver = AdjointSolver(adjoint_params);

% Execute the optimization code
RI_optimized_single=adjoint_solver.solve(current_source,RImap,options);
%% Design case: double plate model
pixel_period = round(thickness_pixel(2)/2);

regularizer_sequence = { ...
    BoundRegularizer(RI_list(1), RI_list(2)), ...
    BinaryRegularizer(RI_list(1), RI_list(2), 0.5, 1, @(step) (step>=50)*(1+(step-50)/20)), ...
    CyclicConv2Regularizer(filter,'z', @(step) 15<=step), ...
};
density_projection_sequence = { ...
    BoundRegularizer(0, 1), ...
    PeriodicAvgRegularizer('z', pixel_period, @(step) 15<=step), ...
    RotSymRegularizer('z',2, @(step) 15<=step), ...
};
grad_weight = 60;

adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 70;
adjoint_params.optimizer = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight);
adjoint_params.verbose = true;
adjoint_params.verbose_level = 0;
adjoint_params.sectioning_axis = "x";

adjoint_solver = AdjointSolver(adjoint_params);

% Execute the optimization code
RI_optimized_double=adjoint_solver.solve(current_source,RImap,options);
%% Visualization
E_field_list = cell(2,1);
for i = 1:2
    if i == 1
        RI_optimized = RI_optimized_single;
        plate_type = "single";
    else
        RI_optimized = RI_optimized_double;
        plate_type = "double";
    end
    forward_solver.set_RI(RI_optimized);
    E_field = forward_solver.solve(current_source);
    E_intensity = sum(abs(E_field),4);
    viewerContinuous = viewer3d(BackgroundColor="white",BackgroundGradient="off",CameraZoom=2);
    hVolumeContinuous = volshow(real(RI_optimized), OverlayData=E_intensity, Parent= viewerContinuous, OverlayAlphamap = linspace(0,0.2,256),...
        OverlayRenderingStyle = "GradientOverlay", RenderingStyle = "GradientOpacity", OverlayColormap=parula);

    center = size(E_intensity)/2;
    viewerContinuous.CameraTarget = center;
    
    final_cost = sum(intensity_weight.*E_intensity,'all');
    fprintf("final cost of %s plate: %g\n\n",plate_type,final_cost)
    E_field_list{i} = E_field;
end
disp("Higher the cost, better the result")

%% Optional: save RI configuration
filename = sprintf('optimized double helix mask_Diameter-%.2fum_single plate.mat',diameter);
save_RI(filename, RI_optimized_single, params.resolution, params.wavelength,E_field_list{1});
filename = sprintf('optimized double helix mask_Diameter-%.2fum_double plate.mat',diameter);
save_RI(filename, RI_optimized_double, params.resolution, params.wavelength,E_field_list{2});

%% Optional Design case: free-form plate model
regularizer_sequence = { ...
    BoundRegularizer(RI_list(1), RI_list(2)), ...
    BinaryRegularizer(RI_list(1), RI_list(2), 0.5, 1, @(step) (step>=50)*(1+(step-50)/20)) ...
};
density_projection_sequence = { ...
    BoundRegularizer(0, 1), ...
    RotSymRegularizer('z',2, @(step) 15<=step), ...
};
grad_weight = 60;

adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 70;
adjoint_params.optimizer = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight);
adjoint_params.verbose = true;
adjoint_params.verbose_level = 0;
adjoint_params.sectioning_axis = "z";
adjoint_params.sectioning_position = thickness_pixel(1)+1;
adjoint_params.temp_save_dir = "temp_double_helix_free_form";

adjoint_solver = AdjointSolver(adjoint_params);

% Execute the optimization code
RI_optimized_free_form=adjoint_solver.solve(current_source,RImap,options);
forward_solver.set_RI(RI_optimized_free_form);
E_field_free_form = forward_solver.solve(current_source);

filename = sprintf('optimized double helix mask_Diameter-%.2fum_free form.mat',diameter);
save_RI(filename, RI_optimized_free_form, params.resolution, params.wavelength,E_field_free_form);

%% Performance comaprision
E_field_list = cell(2,1);
for i = 1:2
    if i == 1
        RI_optimized = RI_optimized_single;
        plate_type = "single";
    else
        RI_optimized = RI_optimized_free_form;
        plate_type = "free_form";
    end
    forward_solver.set_RI(RI_optimized);
    E_field = forward_solver.solve(current_source);
    E_intensity = sum(abs(E_field),4);
    viewerContinuous = viewer3d(BackgroundColor="white",BackgroundGradient="off",CameraZoom=2);
    hVolumeContinuous = volshow(real(RI_optimized), OverlayData=E_intensity, Parent= viewerContinuous, OverlayAlphamap = linspace(0,0.2,256),...
        OverlayRenderingStyle = "GradientOverlay", RenderingStyle = "GradientOpacity", OverlayColormap=parula);

    center = size(E_intensity)/2;
    viewerContinuous.CameraTarget = center;
    
    final_cost = sum(intensity_weight.*E_intensity,'all');
    fprintf("final cost of %s plate: %g\n\n",plate_type,final_cost)
    E_field_list{i} = E_field;
end
disp("Higher the cost, better the result")
