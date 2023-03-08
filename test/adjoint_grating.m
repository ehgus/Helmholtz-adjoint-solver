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
params.size=[101 101 81]; % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FIELD_GENERATOR.get_field(field_generator_params);

%3 phantom RI generation parameter
material_RI = [min_RI 1.5 min_RI];
thickness_pixel = [35 16]; % 350 um, 150 um, ...
RI = phantom_plate(params.size, material_RI, thickness_pixel);

%% forward solver
%forward solver parameters
forward_params=params;
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];
[minRI, maxRI] = bounds(RI,"all");
forward_params.RI_bg = minRI;

%compute the forward field using convergent Born series
forward_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_params);
display_RI_and_E_field(forward_solver,RI,input_field,'before optimization');
%% Adjoint solver
simulation_size = [101 101];
assert(all(simulation_size <= params.size(1:2)), 'simulation must be smaller than RI map');
ROI_change_xy = padarray(ones(simulation_size, 'logical'),floor((params.size(1:2)-simulation_size)/2), false, 'pre');
ROI_change_xy = padarray(ROI_change_xy,ceil((params.size(1:2)-simulation_size)/2), false, 'post');
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = and(real(RI) > 1.45, ROI_change_xy);
adjoint_params.step = 10;
adjoint_params.itter_max = 10;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 150;
adjoint_params.nmin = 1.4;
adjoint_params.nmax = 10;2.87;
adjoint_params.spatial_filtering_count = -1;
adjoint_params.verbose = false;
adjoint_params.averaging_filter = [false true true];

adjoint_solver = ADJOINT_SOLVER(adjoint_params);
%Adjoint field parameter
diffraction_order = struct;
diffraction_order.x = [-3 3];
diffraction_order.y = [0 0];
x_length = diffraction_order.x(2) - diffraction_order.x(1) + 1;
y_length = diffraction_order.y(2) - diffraction_order.y(1) + 1;
relative_intensity = NaN(x_length, y_length, 3);
relative_intensity(:,floor(y_length/2)+1, 1) = sqrt([0 0 0.17 0 0.64 0 0.17]);
ROI_field = [1, 1, sum(thickness_pixel)+1; size(RI)];
options = struct;
options.relative_intensity = relative_intensity;
options.diffraction_order = diffraction_order;
options.ROI_field = ROI_field;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);

% Configuration for optimized metamaterial
display_RI_and_E_field(forward_solver,RI_optimized,input_field,'after optimization')
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
