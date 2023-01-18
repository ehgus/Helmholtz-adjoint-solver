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
min_RI = 1.4;
max_RI = 2.87;

params.NA=1; % Numerical aperture
params.wavelength=0.355; % [um]
params.RI_bg=min_RI; % Background RI
params.resolution=[0.01 0.01 0.01]; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=[100 100 101]; % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FIELD_GENERATOR.get_field(field_generator_params);

%3 phantom RI generation parameter
material_RI = [min_RI max_RI min_RI];
thickness_pixel = [40 1]; % 400 um, 10 um, ...
phantom_params.name=[min_RI max_RI min_RI];
RI = phantom_plate(params.size, material_RI, thickness_pixel);

%% forward solver
%forward solver parameters
forward_params=params;
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];
[minRI, maxRI] = bounds(RI,"all");
forward_params.RI_bg = minRI; double(sqrt((minRI^2+maxRI^2)/2));minRI;

%compute the forward field using convergent Born series
forward_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_params);
display_RI_and_E_field(forward_solver,RI,input_field,'before optimization');
%% Adjoint solver
simulation_size = [81 81];
assert(all(simulation_size <= params.size(1:2)), 'simulation must be smaller than RI map');
ROI_change_xy = padarray(ones(simulation_size, 'logical'),floor((params.size(1:2)-simulation_size)/2), false, 'pre');
ROI_change_xy = padarray(ROI_change_xy,ceil((params.size(1:2)-simulation_size)/2), false, 'post');
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = and(real(RI) > 2, ROI_change_xy);
adjoint_params.step = 0.5;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 100;
adjoint_params.nmin = 1.4;
adjoint_params.nmax = 2.87;
adjoint_params.verbose = true;

adjoint_solver = ADJOINT_SOLVER(adjoint_params);
%Adjoint field parameter
ROI_field = and(1.6 < real(RI), real(RI) < 1.7);
diffraction_order = [-3 -2 -1 0 1 2 3];
relative_intensity = [0 0 0.17 0 0.64 0 0];
transmission_weight = table(diffraction_order, relative_intensity);
options = struct;
options.ROI_field = ROI_field;
options.transmission_weight = transmission_weight;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);

%% utilities
function display_RI_and_E_field(forward_solver,RI,input_field,figureName)
    forward_solver.set_RI(RI); % change to RI_optimized and run if you want to see the output of adjoint method
    tic;
    [field_trans,~,field_3D]=forward_solver.solve(input_field(:,:,:,1));
    toc;

    % tranform vector field to scalar field
    [input_field_scalar,field_trans_scalar]=vector2scalarfield(input_field,field_trans);
    input_field_no_zero=input_field_scalar;
    zero_part_mask=abs(input_field_scalar)<=0.01*mean(abs(input_field_scalar),'all');
    input_field_no_zero(zero_part_mask)=0.01*exp(1i.*angle(input_field_no_zero(zero_part_mask)));
    relative_complex_trans_field = field_trans_scalar./input_field_no_zero;
    intensity_map = sum(abs(field_3D).^2,4);
    
    % Display results: transmitted field
    figure('Name',figureName);colormap parula;
    subplot(2,1,1);imagesc(squeeze(abs(relative_complex_trans_field)));title('Amplitude of transmitted light');
    subplot(2,1,2);imagesc(squeeze(angle(relative_complex_trans_field)));title('Phase of transmitted light');
    figure('Name',[figureName '- intensity map']);orthosliceViewer(intensity_map);title('amplitude in material');colormap gray
    figure('Name',[figureName '- real RI map']);orthosliceViewer(real(RI));title('RI of material');colormap gray
    figure('Name',[figureName '- intensity and RI']);hold on;
    plot(squeeze(real(field_3D(floor(end/2),floor(end/2),:,1))), 'r');
    plot(squeeze(real(RI(floor(end/2),floor(end/2),:))),'b');legend('E field','RI');title('Values along z aixs');
end
