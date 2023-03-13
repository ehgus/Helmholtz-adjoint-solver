clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters
NA=1;
%% load RI profiles
[RI_metalens, resolution, wavelength] = load_RI('optimized_RI.mat');

%% set optical parameters

%0 gpu accelerator
target_gpu_device=1;
gpu_device=gpuDevice(target_gpu_device);
MULTI_GPU=false; % Use Multiple GPU?

%1-0 common optical parameters
params.NA=NA; % Numerical aperture
params.wavelength=wavelength; % [um]
params.resolution=resolution; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=size(RI_metalens); % 3D volume grid
params.return_3D = true;
params.verbose = false;

%% incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FieldGenerator.get_field(field_generator_params);

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
PDMS = material(RefractiveIndexDB(),"organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
params_FDTD.RI_bg=real(PDMS(wavelength));
params_FDTD.is_plane_wave = true;

unit_thickness = 2;
iteration_number = 5;
E_field_rst = cell(1,iteration_number);
for idx = 1:iteration_number
    params_FDTD.boundary_thickness = [0 0 (idx-1)* unit_thickness];
    forward_solver = FDTDsolver(params_FDTD);
    forward_solver.set_RI(RI_metalens);
    [~, ~, E_field_rst{idx}] = forward_solver.solve(input_field);
end

%% display result
intensity_list = arrayfun(@(x)(sum(abs(x{1}).^2,4)), E_field_rst,'UniformOutput',false);

scale = (1:81)*resolution(1);
max_val = max(intensity_list{1},[],'all');
figure;
hold on
for idx = 1:iteration_number
    plot(scale,squeeze(intensity_list{idx}(41,41,:)),'DisplayName',sprintf("%d um padding",(idx-1)* unit_thickness));
end
legend;