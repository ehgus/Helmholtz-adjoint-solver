% Metalens simulator for optimized lens
% The following script needs mat file having a optimized RI profile
% A example profile can be acquired after executing "ADJOINT_EXAMPLE.m"

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters
NA=1;
wavelength = 0.355;
resolution = [1 1 1]*wavelength/10/NA;

%% load RI profiles
RI_metalens = load_RI_data('optimized_RI.h5');

phantom_params=PHANTOM.get_default_parameters();
phantom_params.outer_size = size(RI_metalens);
phantom_params.resolution = resolution;
phantom_params.wavelength = wavelength;
phantom_params.cd0 = dirname;
phantom_params.name=["PDMS","TiO2", "Microchem SU-8 2000"];
phantom_params.thickness = [wavelength 0.15 size(RI_metalens)*resolution(3)];
RI_flat= PHANTOM.get_TiO2_mask(phantom_params); % PDMS, TiO2, SU-8 layered structure 

RI_homogeneous = zeros('like',RI_metalens);
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

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 20];
[minRI, maxRI] = bounds(RI_metalens,"all");
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

%1-2 FDTD parameters
params_FDTD=params;
params_FDTD.use_GPU=false;
params_FDTD.boundary_thickness = [0 0 0];
params_FDTD.RI_bg=real(get_RI(dirname,"PDMS", wavelength));
params_FDTD.is_plane_wave = true;
params_FDTD.PML_boundary = [false false true];

forward_solver_list = { ...
    FORWARD_SOLVER_CONVERGENT_BORN(params_CBS), ...
    FORWARD_SOLVER_FDTD(params_FDTD) ...
};

%% incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
field_generator=FIELD_GENERATOR(field_generator_params);
input_field=field_generator.get_fields();

%% solve the foward problem
RI_type = 'metalens';

E_field_rst = cell(2,1);
for isolver = 1:2
    forward_solver = forward_solver_list{isolver};
    forward_solver.set_RI(RI_patterns.(RI_type));
    [~, ~, E_field_rst{isolver}] = forward_solver.solve(input_field);
end

%% draw results
intensity_list = arrayfun(@(x)(sum(abs(x{1}).^2,4)), E_field_rst,'UniformOutput',false);
concat_intensity = cat(2,intensity_list{:});
figure('Name','Intensity: CBS / FDTD');
orthosliceViewer(concat_intensity);
colormap parula

figure;
subplot(2,2,[1,2]);
imagesc(squeeze(concat_intensity(round(params.size(2)/2),:,:))');
colormap parula;
yline(41,'--yellow');
subplot(2,2,3);
plot(intensity_list{1}(:,41,41));
title('CBS')
subplot(2,2,4);
plot(intensity_list{2}(:,41,41));
title('FDTD')
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
