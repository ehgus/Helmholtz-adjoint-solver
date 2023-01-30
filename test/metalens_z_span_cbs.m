clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% basic optical parameters
NA=1;
wavelength = 0.355;
resolution = [1 1 1]*wavelength/10/NA;

%% load RI profiles
RI_metalens = load_RI_data('optimized_RI_no_pad.h5');
%RI_metalens = padarray(RI_metalens, [0, 0, 10],'replicate','pre');
%RI_metalens = padarray(RI_metalens, [0, 0, 300],'replicate','post');

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

%1-1 CBS parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 50];
%[minRI, maxRI] = bounds(RI,"all");
params_CBS.RI_bg = real(get_RI(RI_DB(),"Microchem SU-8 2000",wavelength));%double(sqrt((minRI^2+maxRI^2)/2))
params_CBS.max_attenuation_width = [0 0 40];

unit_thickness = 2;
iteration_number = 5;
E_field_rst = cell(1,iteration_number);
for idx = 1:iteration_number
    thickness_pixel = round((idx-1)*unit_thickness*params_CBS.wavelength/params_CBS.RI_bg./(params_CBS.resolution.*2)); 
    RI_metalens_pad = padarray(RI_metalens, [0 0 (idx-1)*unit_thickness], 'replicate');
    params_CBS.size = size(RI_metalens_pad);
    forward_solver = ConvergentBornSolver(params_CBS);
    forward_solver.set_RI(RI_metalens_pad);
    [~, ~, E_field_rst{idx}] = forward_solver.solve(input_field);
    E_field_rst{idx} = E_field_rst{idx}(:,:,(1:size(RI_metalens,3))+(idx-1)*unit_thickness,:);
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