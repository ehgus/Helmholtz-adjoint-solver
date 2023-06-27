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
SiO2 = database.material("main","SiO2","Malitson");
params.RI_bg=H2O(params.wavelength);
RI_sp=SiO2(params.wavelength);
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
forward_FDTD_params.use_GPU=false;
forward_FDTD_params.use_cuda=false;
forward_FDTD_params.verbose=false;
forward_FDTD_params.boundary_thickness=[0 0 1];
forward_FDTD_params.iterations_number=-1;
forward_FDTD_params.is_plane_wave = true;
forward_FDTD_params.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

% forward solver parameters - MIE
forward_params_Mie=params;
forward_params_Mie.use_GPU=true;
forward_params_Mie.truncated = true;
forward_params_Mie.verbose = false;
forward_params_Mie.boundary_thickness = [0 0 0];
forward_params_Mie.lmax = 80;
forward_params_Mie.n_s = RI_sp;
forward_params_Mie.radius = radius;
forward_params_Mie.divide_section = 10;

%% create phantom RI and field
% make the phantom
RI=phantom_bead(params.size, [params.RI_bg, RI_sp], round(radius/params.resolution(3)));

%create the incident field
%2 illumination parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';
input_field=FieldGenerator.get_field(field_generator_params);
source_params = params;
source_params.polarization = [1 0 0];
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;
current_source = PlaneSource(source_params);
%% solve the forward problem
cbs_file = "SiO2_5um_bead_CBS.mat";
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

fdtd_file = "SiO2_5um_bead_FDTD.mat";
if isfile(fdtd_file)
    load(fdtd_file, 'field_FDTD', 'Hfield_FDTD');
else
    forward_solver_FDTD=FDTDsolver(forward_FDTD_params);
    forward_solver_FDTD.set_RI(RI);
    tic;
    [field_FDTD, Hfield_FDTD]=forward_solver_FDTD.solve(input_field);
    toc;
    save(fdtd_file, 'field_FDTD', 'Hfield_FDTD');
end
%compute the forward field - Mie
mie_field_filename = fullfile(dirname,'test/Mie_field.mat');
if isfile(mie_field_filename)
    load(mie_field_filename, 'field_3D_Mie');
else
    forward_solver_Mie=MieTheorySolver(forward_params_Mie);
    forward_solver_Mie.set_RI(RI);
    tic;
    [field_3D_Mie]=forward_solver_Mie.solve(input_field);
    toc;
    save(mie_field_filename, 'field_3D_Mie');
end

%% Draw results

% 3D field distribution: E field
intensity_CBS=sum(abs(field_CBS(:,:,:,:,1)).^2,4);
intensity_FDTD=sum(abs(field_FDTD(:,:,:,:,1)).^2,4);
intensity_Mie=sum(abs(field_3D_Mie(:,:,:,:,1)).^2,4);
figure('Name','Intensity: CBS / FDTD / Mie scattering');
orthosliceViewer(cat(2,intensity_CBS,intensity_FDTD,intensity_Mie));
colormap parula;

% 3D field distribution: H field
H_intensity_CBS=sum(abs(Hfield_CBS(:,:,:,:,1)).^2,4);
H_intensity_FDTD=sum(abs(Hfield_FDTD(:,:,:,:,1)).^2,4);
figure('Name','H field Intensity: CBS / FDTD');
orthosliceViewer(cat(2,H_intensity_CBS, H_intensity_FDTD));
colormap parula;

% intensity cross section through the center of the bead
figure('Name','Z-axis intensity profile througth the center of a bead'); hold on;
grid_idx = (0:size(RI,3)-1)*params.resolution(3);
range_of_bead = [-radius radius] + grid_idx(fix(end/2));
line(grid_idx, squeeze(intensity_Mie(round(end/2),round(end/2),:)),'Color', 'black','LineWidth',1);
line(grid_idx, squeeze(intensity_CBS(round(end/2),round(end/2),:)),'Color', '#0072BD','LineWidth',1);
line(grid_idx, squeeze(intensity_FDTD(round(end/2),round(end/2),:)),'Color', '#D95319','LineWidth',1);
xline(range_of_bead(1),'--r','inside bead')
xline(range_of_bead(2),'--r','outside bead')
legend('Mie scattering', 'CBS', 'FDTD'), title('Axial intensity')

% intensity cross section images
color_range = [0 15];
figure('Name', 'cross section of E field: CBS / FDTD / Mie scattering');
subplot(1,3,1);
imagesc(squeeze(intensity_CBS(round(end/2),:,:)), color_range);
colormap parula;
subplot(1,3,2);
imagesc(squeeze(intensity_FDTD(round(end/2),:,:)), color_range);
colormap parula;
subplot(1,3,3);
imagesc(squeeze(intensity_Mie(round(end/2),:,:)), color_range);
colormap parula;

%image quality
fprintf("MSE: CBS = %.3g, FDTD = %.3g\n",immse(intensity_Mie, intensity_CBS),immse(intensity_Mie, intensity_FDTD))
fprintf("PSNR: CBS = %.3g, FDTD = %.3g\n",psnr(intensity_Mie, intensity_CBS),psnr(intensity_Mie, intensity_FDTD))
fprintf("SSIM: CBS = %.3g, FDTD = %.3g\n",ssim(intensity_Mie, intensity_CBS),ssim(intensity_Mie, intensity_FDTD))
