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
focal_length = 5; % unit: micron
z_padding = 1;    % padding along z direction
plate_thickness = 0.15;

% Refractive index profile
RI_database = RefractiveIndexDB();
PDMS = RI_database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = RI_database.material("main","TiO2","Siefke");
substrate = RI_database.material("other","resists","Microchem SU-8 2000");
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
optim_region = real(RImap) > 2;
filter = CyclicConv2Regularizer.conic(round(0.2/resolution));

regularizer_sequence = { ...
    BoundRegularizer(RI_list(1), RI_list(2)), ...
    BinaryRegularizer(RI_list(1), RI_list(2), 0.5, 1, @(step) (step>=40)*(1+(step-40)/10)), ...
    CyclicConv2Regularizer(filter,'z'), ...
};
density_projection_sequence = { ...
    BoundRegularizer(0, 1), ...
    AvgRegularizer('z'), ...
    MirrorSymRegularizer('xy'), ...
    RotSymRegularizer('z',1), ...
};
grad_weight = 0.2;

% Adjoint solver
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 60;
adjoint_params.optimizer = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight);
adjoint_params.verbose = true;
adjoint_params.sectioning_axis = "z";
adjoint_params.sectioning_position = thickness_pixel(1)+1;


adjoint_solver = AdjointSolver(adjoint_params);

% Adjoint design paramters
focal_spot_radius_pixel = 0.63 * wavelength * focal_length/diameter/resolution;
intensity_weight = phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], focal_spot_radius_pixel);
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
options.intensity_weight = intensity_weight;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(current_source,RImap,options);

%% Visualization-1
display_RI_Efield(forward_solver,RI_optimized,current_source,'after optimization')

%% Visualization-2
forward_solver.set_RI(RI_optimized);
E_field = forward_solver.solve(current_source);
E_intensity = sum(abs(E_field),4);
viewerContinuous = viewer3d(BackgroundColor="white",BackgroundGradient="off",CameraZoom=2);
hVolumeContinuous = volshow(real(RI_optimized), OverlayData=E_intensity, Parent= viewerContinuous, OverlayAlphamap = linspace(0,0.2,256),...
    OverlayRenderingStyle = "GradientOverlay", RenderingStyle = "GradientOpacity", OverlayColormap=parula);

%% optional: save RI configuration
filename = sprintf('optimized lens Diameter-%.2fum F-%.2fum.mat',diameter,focal_length);
save_RI(filename, RI_optimized, params.resolution, params.wavelength,E_field);