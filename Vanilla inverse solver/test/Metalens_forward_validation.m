% Metalens simulator for optimized lens
%
% [execution guideline]
% The following script needs mat file having a optimized RI profile
% A example profile can be acquired after executing "ADJOINT_EXAMPLE.m"
% After getting the metalens profile, rename "optimized_RI.h5" to "optimized_RI_no_pad.h5"
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
wavelength = 0.355;
oversampling_rate = 3;
resolution = [1 1 1]*wavelength/10/NA/oversampling_rate;

%% load RI profiles
RI_metalens = load_RI_data('optimized_RI_no_pad.h5');
RI_metalens = imresize3(RI_metalens, oversampling_rate, 'nearest');
phantom_params=PHANTOM.get_default_parameters();
phantom_params.outer_size = size(RI_metalens);
phantom_params.resolution = resolution;
phantom_params.wavelength = wavelength;
phantom_params.cd0 = dirname;
phantom_params.name=["PDMS","TiO2", "Microchem SU-8 2000"];
phantom_params.thickness = [wavelength 0.15 size(RI_metalens)*resolution(3)];
RI_flat= PHANTOM.get_TiO2_mask(phantom_params); % PDMS, TiO2, SU-8 layered structure

RI_homogeneous = zeros(size(RI_metalens),'like',RI_metalens);
RI_homogeneous(:) = real(get_RI(dirname,"PDMS",wavelength));

RI_patterns = struct( ...
    'metalens', RI_metalens, ...
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
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=size(RI_metalens); % 3D volume grid
params.return_3D = true;
params.verbose = false;

%% incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FIELD_GENERATOR.get_field(field_generator_params);

%% solve the foward problem
RI_type = 'metalens';
RI = RI_patterns.(RI_type);

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 5];
[minRI, maxRI] = bounds(RI,"all");
params_CBS.RI_bg = real(get_RI(dirname,"Microchem SU-8 2000",wavelength));
params_CBS.max_attenuation_width = [0 0 5];

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.RI_bg=real(get_RI(dirname,"PDMS", wavelength));
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];
params_FDTD.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

forward_solver_list = { ...
    FORWARD_SOLVER_CONVERGENT_BORN(params_CBS), ...
    FORWARD_SOLVER_FDTD(params_FDTD) ...
};
solver_num = length(forward_solver_list);

E_field_rst = cell(solver_num,1);
for isolver = 1:solver_num
    forward_solver = forward_solver_list{isolver};
    save_title = sprintf("metalens_pattern_%s_oversample_%d.mat",class(forward_solver), oversampling_rate);
    if isfile(save_title)
        load(save_title)
        E_field_rst{isolver} = E_field_3D;
        continue
    end
    forward_solver.set_RI(RI);
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

% MSE value
% Set center of phase to be the same
center_position = floor(size(E_field_rst{1},1:3)/2)+1;
for i = 1:2
    center_field = E_field_rst{i}(center_position(1), center_position(2), center_position(3),1);
    center_field = center_field./abs(center_field);
    E_field_rst{i} = E_field_rst{i}./center_field;
end
MSE_test = mean(abs(E_field_rst{1}-E_field_rst{2}).^2, 'all');
fprintf("MSE test result: %f\n",MSE_test);
%%
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
