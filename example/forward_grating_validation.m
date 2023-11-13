% modulated grating simulator

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));
%% basic optical parameters

oversampling_rate = 1;
% load RI profiles
sim_type_list = ["CBS","2D_FDTD"];
NA=1;
RI_grating_pattern = cell(1,2);
for idx = 1:2
    sim_type = sim_type_list{idx};
    [RI_grating_pattern{idx}, ~, wavelength] = load_RI(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),sprintf('%s_optimized grating.mat',sim_type)));
end
resolution = 0.01;
mask_width = 0.15;
grid_size = [11 100 200];

database = RefractiveIndexDB();
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) real(func(wavelength)), {PDMS TiO2 Microchem_SU8_2000});
RI_list(2) = 0;
thickness_pixel = [0.2 mask_width]/resolution;
background_plate = phantom_plate(grid_size, RI_list, thickness_pixel);
RI_grating = cell(1,2);

figure('Name','grating pattern')
for idx = 1:2
    RI_grating{idx} = background_plate;
    RI_grating{idx}(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = RI_grating{idx}(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) + RI_grating_pattern{idx}(1,:,1);
    %oversampling
    resolution = resolution/oversampling_rate;
    if oversampling_rate < 1
        RI_grating{idx} = imresize3(RI_grating{idx}, oversampling_rate, 'linear');
    elseif oversampling_rate > 1
        RI_grating{idx} = imresize3(RI_grating{idx}, oversampling_rate, 'nearest');
    end

    subplot(2,1,idx)
    imagesc(squeeze(RI_grating_pattern{idx}(1,:,:))', real([PDMS(wavelength), TiO2(wavelength)]));
    colormap gray;
end
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
params.size=size(RI_grating{1}); % 3D volume grid
params.verbose = false;
params.RI_bg = RI_list(1);

%% incident field parameters
source_params = params;
source_params.polarization = [1 0 0];
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;
current_source = PlaneSource(source_params);

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 5];
params_CBS.field_attenuation = [0 0 5];
params_CBS.field_attenuation_sharpness = 1;
params_CBS.potential_attenuation = [0 0 3];
params_CBS.potential_attenuation_sharpness = 0.2;

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];
params_FDTD.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');
params_FDTD.hide_GUI = true;
forward_solver=ConvergentBornSolver(params_CBS);
sim_num = length(sim_type_list);
E_field_rst = cell(1,sim_num);
H_field_rst = cell(1,sim_num);

for idx = 1:sim_num
    save_title = sprintf("grating_pattern_%s_oversample_%d_%s.mat",sim_type_list(idx), oversampling_rate, sim_type);
    if isfile(save_title)
        load(save_title)
        E_field_rst{idx} = E_field_3D;
        H_field_rst{idx} = H_field_3D;
        continue
    end
    forward_solver.set_RI(RI_grating{idx});
    tic;
    [E_field_rst{idx}, H_field_rst{idx}] = forward_solver.solve(current_source);
    E_field_3D = E_field_rst{idx};
    H_field_3D = H_field_rst{idx};
    toc;
    save(save_title, 'E_field_3D','H_field_3D');
end

%% draw results
E_intensity_list = arrayfun(@(x)(sum(abs(x{1}).^2,4)), E_field_rst,'UniformOutput',false);
H_intensity_list = arrayfun(@(x)(sum(abs(x{1}).^2,4)), H_field_rst,'UniformOutput',false);
E_concat_intensity = cat(2,E_intensity_list{:});
H_concat_intensity = cat(2,H_intensity_list{:});
max_E_val = 4.3;
max_H_val = 1.5e-4;

% E intensity 
figure('Name','|E|^2: CBS / FDTD');
orthosliceViewer(E_concat_intensity,'DisplayRange',[0 max_E_val]);
colormap parula

% H intensity
figure('Name','|H|^2: CBS / FDTD');
orthosliceViewer(H_concat_intensity,'DisplayRange',[0 max_H_val]);
colormap parula

center_RI = round(size(E_intensity_list{1},1:2)/2);
scale_xy = (1:size(E_intensity_list{1},1))*resolution;
scale_z = (1:size(E_intensity_list{1},3))*resolution;

figure;
for idx = 1:2
    subplot(2,2,idx);
    imagesc(scale_xy, scale_xy, squeeze(E_intensity_list{idx}(center_RI(1),:,:))',[0 max_E_val]);
    xline(center_RI(1),'--yellow');
    colormap parula;
end
subplot(2,2,[3 4]);
hold on;
plot(scale_z,squeeze(E_intensity_list{1}(center_RI(1),center_RI(2),:)));
plot(scale_z,squeeze(E_intensity_list{2}(center_RI(1),center_RI(2),:)));
legend(sim_type_list(1),sim_type_list(2))
ylim([0 max_E_val]);
% field transmittance for each plane wave mode
target_angle_mode = -4:4;
target_transmission = [0 0 0 0.15 0 0.63 0 0.22 0];
eigen_E_field = cell(1,length(target_transmission));
eigen_H_field = cell(1,length(target_transmission));
relative_transmission_ref = zeros(1,length(target_transmission));

params_CBS.RI_bg = RI_list(3);
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
    eigen_S_ref = 2*real(poynting_vector(eigen_E_field{idx}, eigen_H_field{idx}));
    relative_transmission_ref(idx) = abs(sum(eigen_S_ref(:,:,end,3),'all'));
end

relative_transmission = zeros(1,length(target_transmission));

% numerical view
for idx = 1:length(E_field_rst)
    E_field = E_field_rst{idx};
    H_field = H_field_rst{idx};
    
    for field_idx = 1:length(target_transmission)
        eigen_S = poynting_vector(E_field, eigen_H_field{field_idx}) + poynting_vector(conj(eigen_E_field{field_idx}), conj(H_field));
        relative_transmission(field_idx) = sum(eigen_S(:,:,end,3),'all');
    end
    fprintf("%10s :",sim_type_list(idx));
    disp(abs(relative_transmission./relative_transmission_ref).^2);
    fprintf("       FoM = %f\n\n",sum((abs(relative_transmission./relative_transmission_ref).^2 -target_transmission).^2));
end

% theoretical view
ideal_E_field = zeros(size(E_field_rst{1}),'like',eigen_E_field{1});
ideal_H_field = zeros(size(H_field_rst{1}),'like',eigen_H_field{1});
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
