% modulated grating simulator

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters

oversampling_rate = 1;
% load RI profiles
sim_type = '2D_FDTD'; % CBS, 2D_FDTD
NA=1;
[RI_grating_pattern, ~, wavelength] = load_RI(fullfile(fileparts(matlab.desktop.editor.getActiveFilename),sprintf('%s_optimized grating.mat',sim_type)));
resolution = 0.01;
mask_width = 0.15;
grid_size = [11 100 105];

database = RefractiveIndexDB();
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) real(func(wavelength)), {PDMS TiO2 Microchem_SU8_2000});
RI_list(2) = 0;
thickness_pixel = [0.20 mask_width]/resolution;
RI_grating = phantom_plate(grid_size, RI_list, thickness_pixel);
RI_grating(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = RI_grating(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) + RI_grating_pattern(1,:,1);
%oversampling
resolution = resolution/oversampling_rate;
if oversampling_rate < 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'linear');
elseif oversampling_rate > 1
    RI_grating = imresize3(RI_grating, oversampling_rate, 'nearest');
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
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation = [0 0 4];
params_CBS.potential_attenuation_sharpness = 0.5;

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];
params_FDTD.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

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
    [~, ~, E_field_rst{isolver}, H_field_rst{isolver}] = forward_solver.solve(input_field);
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

% E intensity 
figure('Name','|E|^2: CBS / FDTD');
orthosliceViewer(E_concat_intensity);
colormap parula

% H intensity
figure('Name','|H|^2: CBS / FDTD');
orthosliceViewer(H_concat_intensity);
colormap parula

center_RI = round(size(E_intensity_list{1},1:2)/2);
scale_xy = (1:size(E_intensity_list{1},1))*resolution;
scale_z = (1:size(E_intensity_list{1},3))*resolution;

max_val = max(E_concat_intensity, [], 'all')*1.1;
figure;
for i = 1:2
    subplot(2,2,i);
    imagesc(scale_xy, scale_xy, squeeze(E_intensity_list{i}(center_RI(1),:,:))',[0 max_val]);
    xline(center_RI(1),'--yellow');
    colormap parula;
end
subplot(2,2,[3 4]);
hold on;
plot(scale_z,squeeze(E_intensity_list{1}(center_RI(1),center_RI(2),:)));
plot(scale_z,squeeze(E_intensity_list{2}(center_RI(1),center_RI(2),:)));
legend('CBS','FDTD')
ylim([0 max_val]);

% MSE value
% Set center of phase to be the same
center_position = floor(size(E_field_rst{1},1:3)/2)+1;
for i = 1:2
    center_field = E_field_rst{i}(center_position(1), center_position(2), center_position(3),1);
    center_field = center_field./abs(center_field);
    E_field_rst{i} = E_field_rst{i}./center_field;
end
MSE_test = mean(abs(E_field_rst{1}-E_field_rst{2}).^2, 'all');
fprintf("MSE test result(E): %f\n",MSE_test);
MSE_test = mean(abs(H_field_rst{1}-H_field_rst{2}).^2, 'all');
fprintf("MSE test result(H): %f\n",MSE_test);

% field transmittance for each plane wave mode
target_transmission = [0 0 0.15 0 0.63 0 0.22];
E_field = cell(1,length(target_transmission));
H_field = cell(1,length(target_transmission));
relative_transmission_ref = zeros(1,7);

CBS_forward_solver = forward_solver_list{1};
impedance = 377/CBS_forward_solver.RI_bg;
Nsize = CBS_forward_solver.size + 2*CBS_forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(E_field)
    % E field
    illum_order = i - 3;
    sin_theta = (illum_order-1)*wavelength/(params.size(2)*resolution*CBS_forward_solver.RI_bg);
    cos_theta = sqrt(1-sin_theta^2);
    if illum_order < 1
        illum_order = illum_order + Nsize(2);
    end
    E_field{i} = zeros(Nsize([1 2 4]));
    E_field{i}(1,illum_order,1) = prod(Nsize(1:2));
    E_field{i} = ifft2(E_field{i});
    E_field{i} = CBS_forward_solver.padd_field2conv(E_field{i});
    E_field{i} = fft2(E_field{i});
    E_field{i} = reshape(E_field{i}, [size(E_field{i},1),size(E_field{i},2),1,size(E_field{i},3)]).*CBS_forward_solver.refocusing_util;
    E_field{i} = ifft2(E_field{i});
    E_field{i} = CBS_forward_solver.crop_conv2RI(E_field{i});
    E_field{i} = E_field{i}(CBS_forward_solver.ROI(1):CBS_forward_solver.ROI(2),CBS_forward_solver.ROI(3):CBS_forward_solver.ROI(4),CBS_forward_solver.ROI(5):CBS_forward_solver.ROI(6),:);
    % H field
    H_field{i} = zeros(size(E_field{i}),'like',E_field{i});
    H_field{i}(:,:,:,2) = E_field{i}(:,:,:,1) * cos_theta;
    H_field{i}(:,:,:,3) = E_field{i}(:,:,:,1) * (-sin_theta);
    H_field{i} = H_field{i}/impedance;

    eigen_S_ref = 2*real(poynting_vector(E_field{i}, H_field{i}));
    relative_transmission_ref(i) = abs(sum(eigen_S_ref(:,:,end,3),'all'));
end

for j = 1:2
    field = E_field_rst{j};
    Hfield = H_field_rst{j};
    relative_transmission = zeros(1,length(E_field));
    
    for i = 1:length(E_field)
        eigen_S = poynting_vector(field, H_field{i}) + poynting_vector(conj(E_field{i}), conj(Hfield));
        relative_transmission(i) = sum(eigen_S(:,:,end,3),'all');
    end
    disp(abs(relative_transmission./relative_transmission_ref).^2);
end
%% troubleshooting
E_diff = H_intensity_list{1} - H_intensity_list{2};
imagesc(2*squeeze(E_diff(1,:,:)./(H_intensity_list{1}(1,:,:)+H_intensity_list{2}(1,:,:))),[-1 1])