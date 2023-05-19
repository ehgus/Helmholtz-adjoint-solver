clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% user-defined optical parameter
use_GPU = true; % accelerator option

NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.025;  % unit: micron
diameter = 3;       % unit: micron
focal_length = 1; % unit: micron
z_padding = 0.5;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'
verbose = false;

% refractive index profile
database = RefractiveIndexDB();
if strcmp(substrate_type,'SU8')
    substrate = database.material("other","resists","Microchem SU-8 2000");
    plate_thickness = 0.15;
elseif strcmp(substrate_type, 'air')
    substrate = @(x)1;
    plate_thickness = 0.35;
else
    error(['The substrate "' substrate_type '" is not supported'])
end
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 substrate});
thickness_pixel = round([wavelength plate_thickness (focal_length + z_padding)]/resolution);
diameter_pixel = ceil(diameter/resolution);
RImap = phantom_plate([diameter_pixel diameter_pixel sum(thickness_pixel)], RI_list, thickness_pixel);

%1 optical parameters
params.NA=NA;                   % Numerical aperture
params.wavelength=wavelength;   % unit: micron
[minRI, maxRI] = bounds(RI_list);
params.RI_bg=minRI;            % Background RI
params.resolution=ones(1,3) * resolution;         % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.vector_simulation=true;  % True/false: dyadic/scalar Green's function
params.size=size(RImap);        % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FieldGenerator.get_field(field_generator_params);

%% Forward solver

params_CBS=params;
params_CBS.use_GPU=use_GPU;
params_CBS.return_3D = true;
params_CBS.boundary_thickness = [0 0 4];
params_CBS.field_attenuation = [0 0 4];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(params_CBS);
forward_solver.set_RI(RImap);

% Configuration for bulk material
if verbose
    display_RI_Efield(forward_solver,RImap,input_field,'before optimization')
end
%% Adjoint method
x_pixel_coord = transpose((1:size(RImap,1))-diameter_pixel/2);
y_pixel_coord = (1:size(RImap,2))-diameter_pixel/2;
ROI_change_xy = x_pixel_coord.^2 + y_pixel_coord.^2 < diameter_pixel^2/4;
ROI_change = and(real(RImap) > 2, ROI_change_xy);
forward_solver.return_transmission = false;
forward_solver.return_reflection = false;
%Adjoint solver parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Intensity";
adjoint_params.ROI_change = ROI_change;
adjoint_params.step = 0.5;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 0.5;
adjoint_params.binarization_step = 100;
adjoint_params.spatial_diameter = 0.1;
adjoint_params.spatial_filter_range = [10 Inf];
adjoint_params.nmin = RI_list(1);
adjoint_params.nmax = RI_list(2);
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [false false true];
tic;
adjoint_solver = AdjointLensSolver(adjoint_params);
toc;
focal_spot_radius_pixel = 0.63 * wavelength * focal_length/diameter/resolution;
intensity_weight = phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], focal_spot_radius_pixel);
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
%intensity_weight = intensity_weight + 0.5*phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], lens_radius_pixel/2);
options.intensity_weight = intensity_weight;

% Execute the optimization code
RI_optimized_byproduct=adjoint_solver.solve(input_field,RImap,options);
adjoint_solver.step = 0.1;
adjoint_solver.itter_max = 100;
adjoint_solver.binarization_step = 40;
adjoint_params.spatial_filter_range = [1 Inf];
RI_optimized=adjoint_solver.solve(input_field,RI_optimized_byproduct,options);
% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')

%% optional: save RI configuration
filename = sprintf('optimized lens on %s Diameter-%.2fum F-%.2fum.mat',substrate_type,diameter,focal_length);
save_RI(filename, RI_optimized, params.resolution, params.wavelength);