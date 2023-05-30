 clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% user-defined optical parameter
use_GPU = true; % accelerator option

NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.01; % unit: micron
mask_width = 0.15;
grid_size = [11 100 105];
target_transmission = [0 0 0.15 0 0.63 0 0.22];
verbose = false;

% refractive index profile
database = RefractiveIndexDB();
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) real(func(wavelength)), {PDMS TiO2 Microchem_SU8_2000});
thickness_pixel = [0.20 mask_width]/resolution;
RI = phantom_plate(grid_size, RI_list, thickness_pixel);

%1 optical parameters
params.NA = NA;
params.wavelength = wavelength;
params.RI_bg=RI_list(1);        % Background RI - should be matched with mode decomposition area
params.resolution=ones(1,3) * resolution; % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.vector_simulation=true;  % True/false: dyadic/scalar Green's function
params.size=grid_size;          % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FieldGenerator.get_field(field_generator_params);

%% forward solver

params_CBS=params;
params_CBS.use_GPU=use_GPU;

params_CBS.boundary_thickness = [0 0 5];
params_CBS.field_attenuation = [0 0 5];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation = [0 0 4];
params_CBS.potential_attenuation_sharpness = 0.5;

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(params_CBS);
display_RI_Efield(forward_solver,RI,input_field,'before optimization');
%% Adjoint solver
forward_solver.verbose = true;
ROI_change = zeros(size(RI),'logical');
ROI_change(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = true;
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = ROI_change;
adjoint_params.step = 3;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 150;
adjoint_params.spatial_diameter = 0.1;
adjoint_params.spatial_filter_range = [Inf Inf];
adjoint_params.nmin = RI_list(1);
adjoint_params.nmax = RI_list(2);
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [true false true];

adjoint_solver = AdjointSolver(adjoint_params);
%Adjoint field parameter
options = struct;
options.target_transmission = target_transmission;
options.surface_vector = zeros(adjoint_params.size(1),adjoint_params.size(2),adjoint_params.size(3),3);
options.surface_vector(:,:,end,:) = options.surface_vector(:,:,end,:) + reshape([0 0 1],1,1,1,3);
options.E_field = cell(1,length(options.target_transmission));
options.H_field = cell(1,length(options.target_transmission));

impedance = 377/forward_solver.RI_bg;
Nsize = forward_solver.size + 2*forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(options.E_field)
    % E field
    illum_order = i - 3;
    sin_theta = (illum_order-1)*params.wavelength/(params.size(2)*params.resolution(2)*params.RI_bg);
    cos_theta = sqrt(1-sin_theta^2);
    if illum_order < 1
        illum_order = illum_order + Nsize(2);
    end
    incident_field = zeros(Nsize([1 2 4]));
    incident_field(1,illum_order,1) = prod(Nsize(1:2));
    incident_field = ifft2(incident_field);
    incident_field = forward_solver.padd_field2conv(incident_field);
    incident_field = fft2(incident_field);
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*forward_solver.refocusing_util;
    incident_field = ifft2(incident_field);
    options.E_field{i} = forward_solver.crop_conv2RI(incident_field);
    % H field
    incident_field_H = zeros(Nsize,'like',incident_field);
    incident_field_H(:,:,:,2) = incident_field(:,:,:,1) * cos_theta;
    incident_field_H(:,:,:,3) = incident_field(:,:,:,1) * (-sin_theta);
    incident_field_H = incident_field_H/impedance;
    options.H_field{i} = incident_field_H;
end

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);
% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')

%% optional: save RI configuration
filename = 'CBS_optimized grating.mat';
save_RI(filename, RI_optimized(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))), params.resolution, params.wavelength);