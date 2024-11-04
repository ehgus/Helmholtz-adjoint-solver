clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters

oversampling_rate = 1;
% load RI profiles
target_solver = "CBS";
NA=1;
fname = fullfile(fileparts(matlab.desktop.editor.getActiveFilename),'CBS_optimized grating.mat');
[RI_grating_pattern, ~, wavelength] = load_RI(fname);
steps = -43; % replace grating pattern manually
RI_grating_pattern = circshift(RI_grating_pattern,steps,2);
resolution = 0.01;
mask_width = 0.15;
grid_size = [11 100 200];

database = RefractiveIndexDB();
air = @(~)1.0;
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) func(wavelength), {air PDMS TiO2 Microchem_SU8_2000});
RI_list(end-1) = 0;
thickness_pixel = [1 100 mask_width]/resolution;
grid_size(3) = grid_size(3) + sum(thickness_pixel(1:end-1)) - 20;
background_plate = phantom_plate(grid_size, RI_list, thickness_pixel);

RI_grating = background_plate;
RI_grating(:,:,sum(thickness_pixel(1:end-1))+1:sum(thickness_pixel)) = RI_grating(:,:,sum(thickness_pixel(1:end-1))+1:sum(thickness_pixel)) + RI_grating_pattern(1,:,1);
%oversampling
resolution = resolution/oversampling_rate;
if oversampling_rate < 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'linear');
elseif oversampling_rate > 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'nearest');
end

figure('Name','grating pattern')
imagesc(squeeze(real(RI_grating(1,:,sum(thickness_pixel(1:end-1))+1:sum(thickness_pixel))))', real([PDMS(wavelength), TiO2(wavelength)]));
colormap gray;
figure('Name','grating pattern - ')
orthosliceViewer(real(RI_grating(:,:,end-199:end)))
%% set optical parameters

%0 gpu accelerator
target_gpu_device=1;
gpu_device=gpuDevice(target_gpu_device);
MULTI_GPU=false; % Use Multiple GPU?

%1-0 common optical parameters
params.NA=NA; % Numerical aperture
params.wavelength=wavelength; % [um]
params.resolution=resolution*ones(1,3); % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.size=size(RI_grating); % 3D volume grid
params.verbose = false;
params.RI_bg = RI_list(end-2);

%% incident field parameters
source_params = params;
source_params.polarization = [1 0 0]/1.07;
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;
current_source = PlaneSource(source_params);

%1-1 forward solver parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 3];
params_CBS.field_attenuation = [0 0 3];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation_sharpness = 0.5;
params_CBS.verbose = false;
params_CBS.verbose_level = 1;

forward_solver=ConvergentBornSolver(params_CBS);
forward_solver.set_RI(RI_grating);
tic;
[E_field_rst, H_field_rst] = forward_solver.solve(current_source);
toc;
%% draw results
E_field_sub_rst = E_field_rst(:,:,end-199:end,:);
H_field_sub_rst = H_field_rst(:,:,end-199:end,:);
E_intensity_sub = sum(abs(E_field_sub_rst).^2,4);
H_intensity_sub = sum(abs(H_field_sub_rst).^2,4);
max_E_val = 4.3;
max_H_val = 1.5e-4;

% E intensity 
figure('Name','|E|^2: CBS / FDTD');
orthosliceViewer(E_intensity_sub,'DisplayRange',[0 max_E_val]);
colormap parula

% H intensity
figure('Name','|H|^2: CBS / FDTD');
orthosliceViewer(H_intensity_sub,'DisplayRange',[0 max_H_val]);
colormap parula

center_RI = round(size(E_intensity_sub,1:2)/2);
scale_xy = (1:size(E_intensity_sub,1))*resolution;
scale_z = (1:size(E_intensity_sub,3))*resolution;

figure;
subplot(1,2,1);
imagesc(scale_xy, scale_xy, squeeze(E_intensity_sub(center_RI(1),:,:))',[0 max_E_val]);
xline(center_RI(1),'--yellow');
colormap parula;
subplot(1,2,2);
plot(scale_z,squeeze(E_intensity_sub(center_RI(1),center_RI(2),:)));
legend("CBS")
ylim([0 max_E_val]);
% field transmittance for each plane wave mode
target_angle_mode = -3:3;
target_transmission = [0 0 0.15 0 0.63 0 0.22];
eigen_E_field = cell(1,length(target_transmission));
eigen_H_field = cell(1,length(target_transmission));
relative_transmission_ref = zeros(1,length(target_transmission));

params_CBS.RI_bg = RI_list(end-2);
forward_solver = ConvergentBornSolver(params_CBS);
impedance = 377/forward_solver.RI_bg;
Nsize = forward_solver.size + 2*forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

% 
for idx = 1:length(target_angle_mode)
    illum_order = target_angle_mode(idx);
    k_y = 2*pi*illum_order/(params_CBS.size(2)*params_CBS.resolution(2));
    adj_source_params = params_CBS;
    adj_source_params.polarization = [-1 0 0];
    adj_source_params.direction = 3;
    adj_source_params.horizontal_k_vector = [0 k_y];
    adj_source_params.center_position = [1 1 1];
    adj_source_params.grid_size = adj_source_params.size;
    adj_current_source = PlaneSource(adj_source_params);

    eigen_E_field{idx} = adj_current_source.generate_Efield(zeros(2,3));
    eigen_H_field{idx} = adj_current_source.generate_Hfield(zeros(2,3));
    eigen_E_field{idx} = eigen_E_field{idx}(:,:,end-199:end,:);
    eigen_H_field{idx} = eigen_H_field{idx}(:,:,end-199:end,:);
    eigen_S_ref = 2*real(poynting_vector(eigen_E_field{idx}, eigen_H_field{idx}));
    relative_transmission_ref(idx) = abs(sum(eigen_S_ref(:,:,end,3),'all'));
end

relative_transmission = zeros(1,length(target_transmission));

% numerical view

for field_idx = 1:length(target_transmission)
    eigen_S = poynting_vector(E_field_sub_rst, eigen_H_field{field_idx}) + poynting_vector(conj(eigen_E_field{field_idx}), conj(H_field_sub_rst));
    relative_transmission(field_idx) = sum(eigen_S(:,:,end,3),'all');
end
relative_transmission = relative_transmission./relative_transmission_ref;
fprintf("%10s :","CBS");
disp(abs(relative_transmission).^2);
fprintf("%10s = %f\n\n","FoM",sum((abs(relative_transmission).^2 -target_transmission).^2));

% theoretical view
ideal_E_field = zeros(size(E_field_sub_rst),'like',eigen_E_field{1});
ideal_H_field = zeros(size(H_field_sub_rst),'like',eigen_H_field{1});
relative_phase = sqrt(target_transmission);
for idx = 1:length(target_transmission)
    ideal_E_field = ideal_E_field + relative_phase(idx)*eigen_E_field{idx};
    ideal_H_field = ideal_H_field + relative_phase(idx)*eigen_H_field{idx};
end
figure('Name',"Ideal case");
imagesc(scale_xy, scale_xy, squeeze(sum(abs(ideal_E_field(center_RI(1),:,:,:)),4))',[0 max_E_val]);

for idx = 1:length(target_transmission)
    eigen_S = poynting_vector(ideal_E_field, eigen_H_field{idx}) + poynting_vector(conj(eigen_E_field{idx}), conj(ideal_H_field));
    relative_transmission(idx) = sum(eigen_S(:,:,end,3),'all');
end
fprintf("Ideal case :");
disp(abs(relative_transmission./relative_transmission_ref).^2);
