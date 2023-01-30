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
display_RI_and_E_field(forward_solver,RI,input_field,'before optimization')
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
display_RI_and_E_field(forward_solver,RI_optimized,input_field,'after optimization')

%% optional: save RI configuration
filename = 'optimized_RI.h5';
save_RI_data(filename,real(RI_optimized),params);
RI_test = load_RI_data(filename);

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

function save_RI_data(filename,volume_RI,RI_params)
    % save volume_RI and simulation condition in file_path (HDF5 format)
    if isfile(filename)
        delete(filename)
    end
    h5create(filename,'/RI_final/RI',size(volume_RI));
    h5write(filename,'/RI_final/RI',volume_RI);
    
    attributenames = {'resolution','wavelength'};
    for i = 1:length(attributenames)
        attrname = attributenames{i};
        h5writeatt(filename,'/RI_final',attrname,RI_params.(attrname));
    end
end

function [volume_RI,RI_params] = load_RI_data(filename)
    % save volume_RI and simulation condition in file_path (HDF5 format)
    volume_RI = h5read(filename,'/RI_final/RI');
    RI_params =struct();
    
    attributenames = {'resolution','wavelength'};
    for i = 1:length(attributenames)
        attrname = attributenames{i};
        RI_params.(attrname) = h5readatt(filename,'/RI_final',attrname);
    end
end