 clc, clear;
% load all functions
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% set the simulation parameters

%0 gpu accelerator
target_gpu_device=1;
gpu_device=gpuDevice(target_gpu_device);
MULTI_GPU=false; % Use Multiple GPU?

%1 optical parameters

params.NA=1; % Numerical aperture
params.wavelength=0.355; % [um]
min_RI = 1.4;
max_RI = 2.87;
params.RI_bg=min_RI; % Background RI
params.resolution=[0.01 0.01 0.01]; % 3D Voxel size [um]
params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
params.vector_simulation=true; % True/false: dyadic/scalar Green's function
params.size=[101 101 106]; % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field=FieldGenerator.get_field(field_generator_params);

%3 phantom RI generation parameter
material_RI = [min_RI max_RI min_RI];
thickness_pixel = [0.20 0.17]/params.resolution(1);
RI = phantom_plate(params.size, material_RI, thickness_pixel);
%RI(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = RI(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) + rand([1 101],'single')*(min_RI-max_RI);


%% forward solver
%forward solver parameters
params_CBS=params;
params_CBS.use_GPU=true;
params_CBS.boundary_thickness = [0 0 5];
params_CBS.field_attenuation = [0 0 5];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation = [0 0 2];
params_CBS.potential_attenuation_sharpness = 0.5;
[minRI, maxRI] = bounds(RI,"all"); 
params_CBS.RI_bg = real(minRI);

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(params_CBS);
display_RI_Efield(forward_solver,RI,input_field,'before optimization');
%% Adjoint solver
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = real(RI) > min_RI + 0.01;
adjoint_params.step = 0.1;
adjoint_params.itter_max = 50;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 150;
adjoint_params.nmin = 1.4;
adjoint_params.nmax = 2.87;
adjoint_params.spatial_filtering_count = -1;
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [true false true];

adjoint_solver = AdjointSolver(adjoint_params);
%Adjoint field parameter
options = struct;
options.target_transmission = [0 0 0.17 0 0.64 0 0.17];
options.surface_vector = zeros(adjoint_params.size(1),adjoint_params.size(2),adjoint_params.size(3),3);
options.surface_vector(:,:,end,:) = options.surface_vector(:,:,end,:) + reshape([0 0 1],1,1,1,3);
options.E_field = cell(1,length(options.target_transmission));
options.H_field = cell(1,length(options.target_transmission));

impedance = 377/forward_solver.RI_bg;
Nsize = forward_solver.size + 2*forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(options.E_field)
    % E field
    illum_order = i - 3;
    sin_theta = (illum_order-1)*params.wavelength/(params.size(2)*params.resolution(2)*forward_solver.RI_bg);
    cos_theta = sqrt(1-sin_theta^2);
    if illum_order < 1
        illum_order = illum_order + Nsize(2);
    end
    incident_field = zeros(Nsize([1 2 4]));
    incident_field(1,illum_order,1) = prod(Nsize(1:2));
    incident_field = ifft2(incident_field);
    incident_field = forward_solver.padd_field2conv(incident_field);
    incident_field = fft2(incident_field);
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*forward_solver.refocusing_util;
    incident_field = ifft2(incident_field);
    options.E_field{i} = forward_solver.crop_conv2RI(incident_field);
    % angle correction
    center_angle = options.E_field{i}(51,51,forward_solver.ROI(6)-29,1);
    center_angle = abs(center_angle)/center_angle*exp(1i*1.40291);
    options.E_field{i} = options.E_field{i}*center_angle;
    % H field
    incident_field_H = zeros(Nsize,'like',incident_field);
    incident_field_H(:,:,:,2) = incident_field(:,:,:,1) * 1; cos_theta;
    incident_field_H(:,:,:,3) = incident_field(:,:,:,1) * (-sin_theta);
    incident_field_H = incident_field_H/impedance;
    options.H_field{i} = incident_field_H;
end

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RI,options);

% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,input_field,'after optimization')
%% deocmposition test
field = 0.17*options.E_field{3} + 0.64*options.E_field{5} + 0.17*options.E_field{7};
field = field(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:);
Hfield = 0.17*options.H_field{3} + 0.64*options.H_field{5} + 0.17*options.H_field{7};
Hfield = Hfield(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:);
% same as Hfield = -1i * (forward_solver.wavelength/2/pi)/(120*pi) * forward_solver.curl_field(field);
relative_transmission = zeros(1,7);
relative_transmission_ref = zeros(1,7);
for i = 1:7
    eigen_S = poynting_vector(field, options.H_field{i}(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)) ...
            + poynting_vector(conj(options.E_field{i}(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)), conj(Hfield));
    relative_transmission(i) = sum(eigen_S(:,:,end,3),'all');
    eigen_S_ref = 2*real(poynting_vector(options.E_field{i}(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:), ...
                                         options.H_field{i}(forward_solver.ROI(1):forward_solver.ROI(2),forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)));
    relative_transmission_ref(i) = sum(eigen_S_ref(:,:,end,3),'all');
end
disp(abs(relative_transmission./relative_transmission_ref));

%%
for i = 1:7
    eigen_S_ref = 2*real(poynting_vector(options.E_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(6),:), ...
                                         options.H_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(6),:)));
    disp(sum(eigen_S_ref(:,:,end,3),'all'))
end
