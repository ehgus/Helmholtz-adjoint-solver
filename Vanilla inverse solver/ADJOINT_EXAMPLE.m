clc, clear;
% load all functions
cd0 = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(genpath(cd0));

%% set the simulation parameters

%0 gpu accelerator
target_gpu_device=1;
gpu_device=gpuDevice(target_gpu_device);
MULTI_GPU=false; % Use Multiple GPU?

%1 optical parameters
NA = 1;
wavelength = 0.355;
params=struct(...
    'NA',NA,... % Numerical aperture
    'wavelength',wavelength,... % [um]
    'RI_bg',real(get_RI(cd0,"PDMS", wavelength)),... % Background RI
    'resolution',[1 1 1]*wavelength/10/NA,... % 3D Voxel size [um]
    'size',[81 81 81],... % 3D volume grid
    'use_abbe_sine',false,... % Abbe sine condition according to demagnification condition
    'vector_simulation',true... % True/false: dyadic/scalar Green's function
);
%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
field_generator=FIELD_GENERATOR(field_generator_params);
input_field=field_generator.get_fields();

%3 phantom RI generation parameter
phantom_params=PHANTOM.get_default_parameters();
phantom_params.outer_size = params.size;
phantom_params.resolution = params.resolution;
phantom_params.wavelength = params.wavelength;
phantom_params.cd0 = cd0;
phantom_params.name=["PDMS","TiO2", "Microchem SU-8 2000"];
phantom_params.thickness = [params.wavelength 0.15 params.size(3)*params.resolution(3)];
% create phantom RI of target material (PDMS + TiO2 + Microchem SU-8 2000)
RI = PHANTOM.get_TiO2_mask(phantom_params);

%% Test: Foward solver
% It displays results of light propagation along the target material

%forward solver parameters
forward_params=params;
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.return_transmission = true;
forward_params.boundary_thickness = [0 0 4];
[minRI, maxRI] = bounds(RI,"all");
forward_params.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

%compute the forward field using convergent Born series
forward_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_params);
forward_solver.set_RI(RI); % change to RI_optimized and run if you want to see the output of adjoint method
tic;
[field_trans,~,field_3D]=forward_solver.solve(input_field);
toc;

% Display results: transmitted field
[input_field_scalar,field_trans_scalar]=vector2scalarfield(input_field,field_trans);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));
figure('Name','Amplitude of transmitted light');imagesc(squeeze(abs(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap gray;
figure('Name','Phase of transmitted light');imagesc(squeeze(angle(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap jet;
figure('Name','E field in material');orthosliceViewer(real(field_3D(:,:,:,1)));
figure('Name','Values along z aixs');plot(squeeze(real(field_3D(floor(end/2),floor(end/2),:,1))), 'r');hold on;plot(squeeze(real(RI(floor(end/2),floor(end/2),:,1))),'b');legend('E field','RI');
 %% Adjoint method

%Adjoint solver parameters
adjoint_params = struct(...
    "forward_solver", forward_solver,...
    'mode', 'Intensity'...
);
adjoint_solver = ADJOINT_SOLVER(adjoint_params);

%% Optimization
optimizer_params = struct(...
    "forward_solver", forward_solver,...
    "adjoint_solver", adjoint_solver,...
    "RImin", get_RI(cd0,"PDMS", params.wavelength),...
    "RImax", get_RI(cd0,"TiO2", params.wavelength),...
    "ROI_change",(real(RI) > 2) ...
);
optimizer = TOPOLOGY_OPTIMIZER(optimizer_params);
phantom_params.name = "bead";
phantom_params.inner_size = [5 5 5];
target_intensity = PHANTOM.get(phantom_params);
RI_optimized=optimizer.optimize_RI(input_field,target_intensity,RI);