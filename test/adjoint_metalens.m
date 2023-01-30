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
params.RI_bg=real(get_RI(RI_DB(),"PDMS", params.wavelength)); % Background RI
params.resolution=[1 1 1]*params.wavelength/10/params.NA; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=[81 81 81]; % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FieldGenerator.get_field(field_generator_params);

%3 phantom RI generation parameter
RI_list = get_RI(RI_DB(), ["PDMS","TiO2", "Microchem SU-8 2000"], params.wavelength);
thickness_pixel = round([params.wavelength 0.15]/params.resolution(3));
RI = phantom_plate(params.size, RI_list, thickness_pixel);

%% Test: Forward solver
% It displays results of light propagation along the target material

%forward solver parameters
forward_params=params;
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];
[minRI, maxRI] = bounds(RI,"all");
forward_params.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(forward_params);
forward_solver.set_RI(RI);

% Configuration for bulk material
display_RI_Efield(forward_solver,RI,input_field,'before optimization')
%% Adjoint method
simulation_size = [81 81];
assert(all(simulation_size <= params.size(1:2)), 'simulation must be smaller than RI map');
ROI_change_xy = padarray(ones(simulation_size, 'logical'),floor((params.size(1:2)-simulation_size)/2), false, 'pre');
ROI_change_xy = padarray(ROI_change_xy,ceil((params.size(1:2)-simulation_size)/2), false, 'post');
%Adjoint solver parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Intensity";
adjoint_params.ROI_change = and(real(RI) > 2, ROI_change_xy);
adjoint_params.step = 0.1;
adjoint_params.itter_max = 200;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 50;
adjoint_params.spatial_diameter = 0.2;
adjoint_params.nmin = get_RI(RI_DB(), "PDMS", params.wavelength);
adjoint_params.nmax = get_RI(RI_DB(), "TiO2", params.wavelength);
adjoint_params.verbose = true;

adjoint_solver = AdjointSolver(adjoint_params);
options.intensity_weight  = phantom_bead(params.size, [0 1], 2.5);

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);

% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')

%% optional: save RI configuration
filename = 'optimized_RI.mat';
save_RI(filename, RI_optimized, params.resolution, params.wavelength);