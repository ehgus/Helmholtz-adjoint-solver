clc, clear;
% load all functions
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% set the simulation parameters

%0 gpu accelerator
target_gpu_device=1;
gpu_device=gpuDevice(target_gpu_device);
MULTI_GPU=false; % Use Multiple GPU?

%1 optical parameters

params.NA=1; % Numerical aperture
params.wavelength=0.355; % [um]
min_RI = 1.4;
max_RI = 2.87;
params.RI_bg=min_RI; % Background RI
params.resolution=[0.01 0.01 0.01]; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=[101 101 81]; % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FieldGenerator.get_field(field_generator_params);

%3 phantom RI generation parameter
material_RI = [min_RI min_RI min_RI];
thickness_pixel = [0.25 0.15]/params.resolution(1); % 350 um, 150 um, ...
RI = phantom_plate(params.size, material_RI, thickness_pixel);
RI(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = RI(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) + rand([1 101],'single')*(max_RI-min_RI);


%% forward solver
%forward solver parameters
forward_params=params;
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];

[minRI, maxRI] = bounds(RI,"all");
forward_params.RI_bg = minRI;

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(forward_params);
display_RI_Efield(forward_solver,RI,input_field,'before optimization');
%% Adjoint solver
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = real(RI) > min_RI + 0.01;
adjoint_params.step = 1;
adjoint_params.itter_max = 50;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 150;
adjoint_params.nmin = 1.4;
adjoint_params.nmax = 2.87;
adjoint_params.spatial_filtering_count = -1;
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [true false true];

adjoint_solver = AdjointSolver(adjoint_params);
%Adjoint field parameter
diffraction_order = struct;
diffraction_order.x = [0 0];
diffraction_order.y = [-3 3];
x_length = diffraction_order.x(2) - diffraction_order.x(1) + 1;
y_length = diffraction_order.y(2) - diffraction_order.y(1) + 1;
relative_intensity = NaN(x_length, y_length, 3);
relative_intensity(floor(x_length/2)+1,:, 1) = sqrt([0 0 0.17 0 0.64 0 0.17]);
ROI_field = [1, 1, sum(thickness_pixel)+1; size(RI)];
options = struct;
options.relative_intensity = relative_intensity;
options.diffraction_order = diffraction_order;
options.ROI_field = ROI_field;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);

% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')
%% compare with the result from adjoint FDTD solver
RI_grating_CBS = RI_optimized(1,:,thickness_pixel(1)+1);
[RI_grating, ~, ~] = load_RI('modulated_grating.mat');
figure;
plot(RI_grating)
hold on
plot(RI_grating_CBS)
