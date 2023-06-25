% modulated grating simulator

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));
%% basic optical parameters

oversampling_rate = 2;
% load RI profiles
sim_type = 'CBS'; % CBS, 2D_FDTD
NA=1;
[RI_grating_pattern, ~, wavelength] = load_RI(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),sprintf('%s_optimized grating.mat',sim_type)));
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
RI_grating = phantom_plate(grid_size, RI_list, thickness_pixel);
RI_grating(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = RI_grating(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) + RI_grating_pattern(1,:,1);
%oversampling
resolution = resolution/oversampling_rate;
if oversampling_rate < 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'linear');
elseif oversampling_rate > 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'nearest');
end

figure('Name','grating pattern')
imagesc(squeeze(RI_grating_pattern(1,:,:))', real([PDMS(wavelength), TiO2(wavelength)]));
colormap gray;
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
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=size(RI_grating); % 3D volume grid
params.return_3D = true;
params.verbose = false;
params.RI_bg = RI_list(1);

%% incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FieldGenerator.get_field(field_generator_params);

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
forward_solver_list = { ...
    ConvergentBornSolver(params_CBS), ...
    FDTDsolver(params_FDTD) ...
};
solver_num = length(forward_solver_list);

E_field_rst = cell(solver_num,1);
H_field_rst = cell(solver_num,1);

for isolver = 1:solver_num
    forward_solver = forward_solver_list{isolver};
    save_title = sprintf("grating_pattern_%s_oversample_%d_%s.mat",class(forward_solver), oversampling_rate, sim_type);
    if isfile(save_title)
        load(save_title)
        E_field_rst{isolver} = E_field_3D;
        H_field_rst{isolver} = H_field_3D;
        continue
    end
    forward_solver.set_RI(RI_grating);
    tic;
    [E_field_rst{isolver}, H_field_rst{isolver}] = forward_solver.solve(input_field);
    E_field_3D = E_field_rst{isolver};
    H_field_3D = H_field_rst{isolver};
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
for i = 1:2
    subplot(2,2,i);
    imagesc(scale_xy, scale_xy, squeeze(E_intensity_list{i}(center_RI(1),:,:))',[0 max_E_val]);
    xline(center_RI(1),'--yellow');
    colormap parula;
end
subplot(2,2,[3 4]);
hold on;
plot(scale_z,squeeze(E_intensity_list{1}(center_RI(1),center_RI(2),:)));
plot(scale_z,squeeze(E_intensity_list{2}(center_RI(1),center_RI(2),:)));
legend('CBS','FDTD')
ylim([0 max_E_val]);
% field transmittance for each plane wave mode
target_angle_mode = -4:4;
target_transmission = [0 0 0 0.15 0 0.63 0 0.22 0];
eigen_E_field = cell(1,length(target_transmission));
eigen_H_field = cell(1,length(target_transmission));
relative_transmission_ref = zeros(1,length(target_transmission));

params_CBS.RI_bg = RI_list(3);
CBS_forward_solver = ConvergentBornSolver(params_CBS);
impedance = 377/CBS_forward_solver.RI_bg;
Nsize = CBS_forward_solver.size + 2*CBS_forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(target_angle_mode)
    % E field
    illum_order = target_angle_mode(i);
    sin_theta = illum_order*CBS_forward_solver.wavelength/(CBS_forward_solver.size(2)*CBS_forward_solver.resolution(2)*CBS_forward_solver.RI_bg);
    cos_theta = sqrt(1-sin_theta^2);
    if illum_order < 0
        illum_order = illum_order + Nsize(2);
    end
    incident_field = zeros(Nsize([1 2 4]));
    incident_field(1,illum_order + 1,1) = prod(Nsize(1:2));
    incident_field = ifft2(incident_field);
    incident_field = CBS_forward_solver.padd_field2conv(incident_field);
    incident_field = fft2(incident_field);
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*CBS_forward_solver.refocusing_util;
    incident_field = ifft2(incident_field);
    eigen_E_field{i} = CBS_forward_solver.crop_conv2RI(incident_field);
    % H field
    incident_field_H = zeros(Nsize,'like',incident_field);
    incident_field_H(:,:,:,2) = incident_field(:,:,:,1) * cos_theta;
    incident_field_H(:,:,:,3) = incident_field(:,:,:,1) * (-sin_theta);
    incident_field_H = incident_field_H/impedance;
    eigen_H_field{i} = incident_field_H;
    
    eigen_S_ref = 2*real(poynting_vector(eigen_E_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:), eigen_H_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:)));
    relative_transmission_ref(i) = abs(sum(eigen_S_ref(:,:,end,3),'all'));
end


relative_transmission = zeros(1,length(target_transmission));
for j = 1:length(E_field_rst)
    E_field = E_field_rst{j};
    H_field = H_field_rst{j};
    
    for i = 1:length(target_transmission)
        eigen_S = poynting_vector(E_field, eigen_H_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:)) +...
            poynting_vector(conj(eigen_E_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:)), conj(H_field));
        relative_transmission(i) = sum(eigen_S(:,:,end,3),'all');
    end
    fprintf("%s :",class(forward_solver_list{j}));
    disp(abs(relative_transmission./relative_transmission_ref).^2);
end

% theoretical view
ideal_E_field = zeros(size(E_field_rst{1}),'like',eigen_E_field{1});
ideal_H_field = zeros(size(H_field_rst{1}),'like',eigen_H_field{1});
relative_phase = sqrt(target_transmission);
for i = 1:length(target_transmission)
    ideal_E_field = ideal_E_field + relative_phase(i)*eigen_E_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:);
    ideal_H_field = ideal_H_field + relative_phase(i)*eigen_H_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:);
end
figure('Name',"Ideal case");
imagesc(scale_xy, scale_xy, squeeze(sum(abs(ideal_E_field(center_RI(1),:,:,:)),4))',[0 max_E_val]);

for i = 1:length(target_transmission)
    eigen_S = poynting_vector(ideal_E_field, eigen_H_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:)) +...
        poynting_vector(conj(eigen_E_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:)), conj(ideal_H_field));
    relative_transmission(i) = sum(eigen_S(:,:,end,3),'all');
end
fprintf("Ideal case :");
disp(abs(relative_transmission./relative_transmission_ref).^2);





