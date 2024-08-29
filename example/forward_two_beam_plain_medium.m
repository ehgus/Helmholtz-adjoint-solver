% forward validation by simulating 5um SiO2 bead in water

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));%% set the simulation parameters
%% 1 optical parameters
radius = 1;
params.NA=1.2;
database = RefractiveIndexDB();
params.wavelength=0.532;
H2O = database.material("main","H2O","Daimon-20.0C");
params.RI_bg=H2O(params.wavelength);
RI_sp = H2O(params.wavelength);
params.resolution=ones(1,3) * 0.05;
params.use_abbe_sine=true;
params.size= [201 201 191];

% forward solver parameters - CBS
forward_CBS_params=params;
forward_CBS_params.use_GPU=true;
forward_CBS_params.boundary_thickness=[0 0 6];
forward_CBS_params.field_attenuation=[0 0 6];
forward_CBS_params.verbose=false;
forward_CBS_params.iterations_number=-1;

% forward solver parameters - FDTD
forward_FDTD_params=params;
forward_FDTD_params.verbose=false;
forward_FDTD_params.hide_GUI = false;
forward_FDTD_params.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

%% create phantom RI and field
% make the phantom
RI=phantom_bead(params.size, [params.RI_bg, RI_sp], round(radius/params.resolution(3)));

%2 illumination parameters
illum_order = 3;
ky = 2*pi*illum_order/(params.size(2)*params.resolution(2));
source_params = params;
source_params.polarization = [1 0 0];
source_params.direction = 3;
source_params.horizontal_k_vector = [0 ky];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;
current_source = PlaneSource(source_params);
source_params.horizontal_k_vector = [0 -ky];
current_source(2) = PlaneSource(source_params);

%% solve the forward problem
cbs_file =fullfile(dirname,"example/plain_CBS_two_beam.mat");
if isfile(cbs_file)
    load(cbs_file, 'field_CBS', 'Hfield_CBS');
else
    forward_solver_CBS=ConvergentBornSolver(forward_CBS_params);
    forward_solver_CBS.set_RI(RI);
    tic;
    [field_CBS, Hfield_CBS]=forward_solver_CBS.solve(current_source);
    toc;
    save(cbs_file, 'field_CBS', 'Hfield_CBS');
end

fdtd_file =fullfile(dirname,"example/plain_FDTD_two_beam.mat");
if isfile(fdtd_file)
    load(fdtd_file, 'field_FDTD', 'Hfield_FDTD');
else
    forward_solver_FDTD=FDTDsolver(forward_FDTD_params);
    forward_solver_FDTD.set_RI(RI);
    tic;
    [field_FDTD, Hfield_FDTD]=forward_solver_FDTD.solve(current_source);
    toc;
    save(fdtd_file, 'field_FDTD', 'Hfield_FDTD');
end

%% Draw results

% 3D field distribution: E field
intensity_CBS=sum(abs(field_CBS(:,:,:,:,1)).^2,4);
intensity_FDTD=single(sum(abs(field_FDTD(:,:,:,:,1)).^2,4));
figure('Name','Intensity: CBS / FDTD');
orthosliceViewer(cat(2,intensity_CBS,intensity_FDTD));
colormap parula;

% 3D field distribution: H field
H_intensity_CBS=sum(abs(Hfield_CBS(:,:,:,:,1)).^2,4);
H_intensity_FDTD=sum(abs(Hfield_FDTD(:,:,:,:,1)).^2,4);
figure('Name','H field Intensity: CBS / FDTD');
orthosliceViewer(cat(2,H_intensity_CBS, H_intensity_FDTD));
colormap parula;

% intensity cross section through the center of the bead
figure('Name','Y-axis intensity profile'); hold on;
grid_idx = (0:size(RI,1)-1)*params.resolution(3);
range_of_bead = [-radius radius] + grid_idx(fix(end/2));
line(grid_idx, squeeze(intensity_CBS(round(end/2),:,round(end/2))),'Color', '#0072BD','LineWidth',1);
line(grid_idx, squeeze(intensity_FDTD(round(end/2),:,round(end/2))),'Color', '#D95319','LineWidth',1);
legend('CBS', 'FDTD'), title('lateral intensity')

% intensity cross section images
color_range = [0 15];
figure('Name', 'cross section of E field: CBS / FDTD');
subplot(1,2,1);
imagesc(squeeze(intensity_CBS(round(end/2),:,:)), color_range);
colormap parula;
subplot(1,2,2);
imagesc(squeeze(intensity_FDTD(round(end/2),:,:)), color_range);
colormap parula;
