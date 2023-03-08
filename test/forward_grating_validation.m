% modulated grating simulator

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters
NA=1;
oversampling_rate = 2;
%% load RI profiles
[RI_grating, ~, wavelength] = load_RI('modulated_grating.mat');
resolution = [0.01 0.01 0.01];
RI_bg = 1.4;
RI_grating = padarray(reshape(RI_grating,1,[]),[numel(RI_grating)-1 0 15],'replicate','post');
RI_grating = padarray(RI_grating,[0 0 10],RI_bg,'pre');
RI_grating = padarray(RI_grating,[0 0 80],RI_bg,'post');
resolution = resolution/oversampling_rate;
RI_grating = imresize3(RI_grating, oversampling_rate, 'nearest');

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
params.size=size(RI_grating); % 3D volume grid
params.return_3D = true;
params.verbose = false;

%% incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FieldGenerator.get_field(field_generator_params);

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 5];
params_CBS.RI_bg = RI_bg;
params_CBS.max_attenuation_width = [0 0 0];

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.RI_bg= RI_bg;
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];
params_FDTD.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

forward_solver_list = { ...
    ConvergentBornSolver(params_CBS), ...
    FDTDsolver(params_FDTD) ...
};
solver_num = length(forward_solver_list);

E_field_rst = cell(solver_num,1);
for isolver = 1:solver_num
    forward_solver = forward_solver_list{isolver};
    save_title = sprintf("grating_pattern_%s_oversample_%d.mat",class(forward_solver), oversampling_rate);
    if isfile(save_title)
        load(save_title)
        E_field_rst{isolver} = E_field_3D;
        continue
    end
    forward_solver.set_RI(RI_grating);
    tic;
    [~, ~, E_field_rst{isolver}] = forward_solver.solve(input_field);
    E_field_3D = E_field_rst{isolver};
    toc;
    save(save_title, 'E_field_3D');
end

%% draw results
intensity_list = arrayfun(@(x)(sum(abs(x{1}).^2,4)), E_field_rst,'UniformOutput',false);
concat_intensity = cat(2,intensity_list{:});
figure('Name','Intensity: CBS / FDTD');
orthosliceViewer(concat_intensity);
colormap parula
center_RI = round(size(intensity_list{1},1:2)/2);
scale_xy = (1:size(intensity_list{1},1))*resolution(1);
scale_z = (1:size(intensity_list{1},3))*resolution(3);

max_val = max(concat_intensity, [], 'all')*1.1; 
figure;
for i = 1:2
    subplot(2,2,i);
    imagesc(scale_xy, scale_xy, squeeze(intensity_list{i}(center_RI(1),:,:))',[0 max_val]);
    xline(center_RI(1),'--yellow');
    colormap parula;
end
subplot(2,2,[3 4]);
hold on;
plot(scale_z,squeeze(intensity_list{1}(center_RI(1),center_RI(2),:)));
plot(scale_z,squeeze(intensity_list{2}(center_RI(1),center_RI(2),:)));
legend('CBS','FDTD')
ylim([0 max_val]);