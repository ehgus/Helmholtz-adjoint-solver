clc;clear;close all
current_dirname = fileparts(matlab.desktop.editor.getActiveFilename);
main_dirname = fileparts(fileparts(current_dirname));
addpath(genpath(fullfile(main_dirname)))

%% Basic optical parameters

% user-defined optical parameter
use_GPU = true; % accelerator option

NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.025; % unit: micron
diameter = 3;       % unit: micron
focal_length = 1.5; % unit: micron
z_padding = 0.5;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'
verbose = false;

% refractive index profile
database = RefractiveIndexDB();
% !!! substrate: air, TiO2: pillar, hollow region: PDMS
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");

if strcmp(substrate_type,'SU8')
    substrate = database.material("other","resists","Microchem SU-8 2000");
    plate_thickness = 0.15;
elseif strcmp(substrate_type, 'air')
    substrate = @(x)1;
    plate_thickness = 0.35;
else
    error(['The substrate "' substrate_type '" is not supported'])
end
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 substrate});
thickness_pixel = round([0.25 plate_thickness (focal_length + 0.5)]/resolution);
diameter_pixel = round(diameter/resolution);
RImap = phantom_plate([diameter_pixel,diameter_pixel,sum(thickness_pixel)], ...
    RI_list, thickness_pixel);

% set parameter struct
params.NA=NA; % Numerical aperture
params.wavelength=wavelength; % [um]
params.resolution=resolution*[1 1 1]; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.size=size(RImap); % 3D volume grid
params.verbose = false;

% incident field
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FieldGenerator.get_field(field_generator_params);

%% forward solver

params_CBS = params;
params_CBS.use_GPU = use_GPU;
params_CBS.boundary_thickness = [0 0 5];
params_CBS.field_attenuation = [0 0 5];
params_CBS.field_attenuation_sharpness = 0.5;
[minRI, maxRI] = bounds(RImap,"all");
params_CBS.RI_bg = minRI;

forward_solver = ConvergentBornSolver(params_CBS);
if verbose
    display_RI_Efield(forward_solver,RImap,input_field,'before optimization')
end
%% adjoint solver
x_pixel_coord = transpose((1:size(RImap,1))-diameter_pixel/2);
y_pixel_coord = (1:size(RImap,2))-diameter_pixel/2;
ROI_change_xy = x_pixel_coord.^2 + y_pixel_coord.^2 < diameter_pixel^2/4;
ROI_change = and(real(RImap) > 2, ROI_change_xy);
%Adjoint solver parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Intensity";
adjoint_params.ROI_change = ROI_change;
adjoint_params.step = 10;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 15;
adjoint_params.spatial_diameter = 0.1;
adjoint_params.spatial_filtering_count = -1;
adjoint_params.nmin = PDMS(params.wavelength);
adjoint_params.nmax = TiO2(params.wavelength);
adjoint_params.averaging_filter = [false false true];
adjoint_params.verbose = true;

adjoint_solver = AdjointLensSolver(adjoint_params);
lens_radius_pixel = 0.61*wavelength/RI_list(3)/sin(atan(diameter/2/focal_length))/resolution;
phantom_bead_in_air = phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], lens_radius_pixel);
phantom_bead_in_air = phantom_bead_in_air + phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], lens_radius_pixel/2);
options.intensity_weight = padarray(phantom_bead_in_air,[0 0 sum(thickness_pixel)-size(phantom_bead_in_air,3)], 0,'pre');
% execution
RI_optimized=adjoint_solver.solve(input_field,RImap,options);

%% save the optimized pattern
save(fullfile(current_dirname,sprintf('optimized_UVlens_Diameter_%dum_FocalLength_%dum_%s.mat',diameter,focal_length,substrate_type)),'RI_optimized')
%% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')
