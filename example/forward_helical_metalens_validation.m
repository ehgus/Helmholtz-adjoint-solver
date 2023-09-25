% Metalens simulator for optimized lens
%
% [execution guideline]
% The following script needs mat file having a optimized RI profile
% A example profile can be acquired after executing "ADJOINT_EXAMPLE.m"
% After getting the metalens profile, rename "optimized_RI.mat" to "optimized_RI.mat"
%
% [Result interpretation]
% Note that the FDTD result will shows a weak light focus compared to the counterpart of CBS when oversampling_rate = 1
% It is expected because FDTD does not have enough grid to simulate accurately
% To get more precise results, you need to increase oversampling rate to at least 3

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters
NA=1;
oversampling_rate = 2;
%% load RI profiles
pattern_idx = 0; %MUST CHANGE%
[RI_metalens, resolution, wavelength] = load_RI(dir(sprintf('0%d_optimized*.mat',pattern_idx)).name);
resolution = resolution/oversampling_rate;
if oversampling_rate < 1
    RI_metalens = imresize3(RI_metalens, oversampling_rate, 'linear');
elseif oversampling_rate > 1
    RI_metalens = imresize3(RI_metalens, oversampling_rate, 'nearest');
end

database = RefractiveIndexDB();
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 PDMS});

thickness_pixel = round([wavelength 0.15]/resolution(3));
RI_flat = phantom_plate(size(RI_metalens), RI_list, thickness_pixel);
RI_flat = real(RI_flat);

RI_homogeneous = zeros(size(RI_metalens),'like',RI_metalens);
RI_homogeneous(:) = real(PDMS(wavelength));

RI_type = sprintf('helical_metalens_0%d',pattern_idx);

RI_patterns = struct( ...
    RI_type, RI_metalens, ...
    'flat', RI_flat, ...
    'homogen', RI_homogeneous ...
);

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
params.size=size(RI_metalens); % 3D volume grid
params.verbose = false;

%% incident field parameters
source_params = params;
source_params.polarization = [1 -1i 0]/sqrt(2);
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;

current_source = PlaneSource(source_params);

%% solve the forward problem

RI = RI_patterns.(RI_type);
[minRI, maxRI] = bounds(RI,"all");

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 3];
params_CBS.field_attenuation = [0 0 3];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = minRI;

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.RI_bg = real(minRI);
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];
params_FDTD.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');
params_FDTD.hide_GUI = false;

forward_solver_list = { ...
    ConvergentBornSolver(params_CBS), ...
    FDTDsolver(params_FDTD) ...
};
solver_num = length(forward_solver_list);

E_field_rst = cell(solver_num,1);
H_field_rst = cell(solver_num,1);

for isolver = 1:solver_num
    forward_solver = forward_solver_list{isolver};
    save_title = sprintf("%s_pattern_%s_oversample_%d.mat",RI_type, class(forward_solver), oversampling_rate);
    if isfile(save_title)
        load(save_title)
        E_field_rst{isolver} = E_field_3D;
        H_field_rst{isolver} = H_field_3D;
        continue
    end
    forward_solver.set_RI(RI);
    tic;
    [E_field_rst{isolver}, H_field_rst{isolver}] = forward_solver.solve(current_source);
    toc;
    E_field_3D = E_field_rst{isolver};
    H_field_3D = H_field_rst{isolver};
    save(save_title, 'E_field_3D', 'H_field_3D');
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
scale_xy = (1:size(E_intensity_list{1},1))*resolution(1);
scale_z = (1:size(E_intensity_list{1},3))*resolution(3);

max_val = max(E_concat_intensity, [], 'all')*1.1;
figure;
for idx = 1:2
    subplot(2,2,idx);
    imagesc(scale_xy, scale_xy, squeeze(E_intensity_list{idx}(center_RI(1),:,:))',[0 max_val]);
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
for idx = 1:2
    center_field = E_field_rst{idx}(center_position(1), center_position(2), center_position(3),1);
    center_field = center_field./abs(center_field);
    E_field_rst{idx} = E_field_rst{idx}./center_field;
end
MSE_test = mean(abs(E_field_rst{1}-E_field_rst{2}).^2, 'all');
fprintf("MSE test result(E): %f\n",MSE_test);
MSE_test = mean(abs(H_field_rst{1}-H_field_rst{2}).^2, 'all');
fprintf("MSE test result(H): %f\n",MSE_test);

