% forward validation by simulating 5um SiO2 bead in water

clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));%% set the simulation parameters
%% 1 optical parameters
radius = 1.5;
params.NA=1.2;
database = RefractiveIndexDB();
params.wavelength=0.532;
H2O = database.material("main","H2O","Daimon-20.0C");
SiO2 = database.material("main","SiO2","Malitson");
params.RI_bg=H2O(params.wavelength);
RI_sp=SiO2(params.wavelength);
params.resolution=[1 1 1]*params.wavelength/8/params.NA;
params.vector_simulation=true;
params.use_abbe_sine=true;
params.size= [161 161 81];
%params.size=[501 501 81];
%params.size= [171 171 91];

% forward solver parameters - CBS
forward_CBS_params=params;
forward_CBS_params.use_GPU=true;
forward_CBS_params.return_3D=true;
forward_CBS_params.return_reflection=true;
forward_CBS_params.boundary_thickness=[0 0 6];%[1 1 2];
forward_CBS_params.field_attenuation=[0 0 6];
forward_CBS_params.verbose=false;
forward_CBS_params.iterations_number=-1;

% forward solver parameters - FDTD
forward_FDTD_params=params;
forward_FDTD_params.use_GPU=false;
forward_FDTD_params.use_cuda=false;
forward_FDTD_params.return_3D=true;
forward_FDTD_params.return_reflection=true;
forward_FDTD_params.verbose=false;
forward_FDTD_params.boundary_thickness=[0 0 38];
forward_FDTD_params.iterations_number=-1;
forward_FDTD_params.is_plane_wave = true;
forward_FDTD_params.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

% forward solver parameters - MIE
forward_params_Mie=params;
forward_params_Mie.use_GPU=true;
forward_params_Mie.truncated = true;
forward_params_Mie.verbose = false;
forward_params_Mie.boundary_thickness = [0 0 0];
forward_params_Mie.return_3D = true;
forward_params_Mie.lmax = 80;
forward_params_Mie.n_s = RI_sp;
forward_params_Mie.radius = 1.5;
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

%% solve the forward problem
%compute the forward field
forward_solver_CBS=ConvergentBornSolver(forward_CBS_params);
forward_solver_CBS.set_RI(RI);
tic;
[field_trans_CBS,field_ref_CBS,field_CBS,Hfield_CBS]=forward_solver_CBS.solve(input_field);
toc;

fdtd_file = "SiO2_5um_bead_FDTD.mat";
if isfile(fdtd_file)
    load field_trans_FDTD field_ref_FDTD field_FDTD Hfield_FDTD
else
    forward_solver_FDTD=FDTDsolver(forward_FDTD_params);
    forward_solver_FDTD.set_RI(RI);
    tic;
    [field_trans_FDTD,field_ref_FDTD,field_FDTD,Hfield_FDTD]=forward_solver_FDTD.solve(input_field);
    toc;
    save(fdtd_file,'field_trans_FDTD','field_ref_FDTD','field_FDTD','Hfield_FDTD');
end
%compute the forward field - Mie
mie_field_filename = fullfile(dirname,'test/Mie_field.mat');
if isfile(mie_field_filename)
    load(mie_field_filename)
else
    forward_solver_Mie=MieTheorySolver(forward_params_Mie);
    forward_solver_Mie.set_RI(RI);
    tic;
    [field_trans_Mie,field_ref_Mie,field_3D_Mie]=forward_solver_Mie.solve(input_field);
    toc;
    save(mie_field_filename, 'field_trans_Mie','field_ref_Mie','field_3D_Mie','-v7.3')
end

%% Draw results
[~,field_trans_CBS_scalar]=vector2scalarfield(input_field,field_trans_CBS);
[~,field_trans_FDTD_scalar]=vector2scalarfield(input_field,field_trans_FDTD);
[~,field_ref_CBS_scalar]=vector2scalarfield(input_field,field_ref_CBS);
[input_field_scalar,field_ref_FDTD_scalar]=vector2scalarfield(input_field,field_ref_FDTD);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));

amp_CBS=squeeze(abs(field_trans_CBS_scalar(:,:,:).^2));
ref_CBS=squeeze(abs(field_ref_CBS_scalar(:,:,:).^2));
ang_CBS=squeeze(angle(field_trans_CBS_scalar(:,:,:)./input_field_no_zero(:,:,:)/field_trans_CBS_scalar(1,1,1)));

amp_FDTD=squeeze(abs(field_trans_FDTD_scalar(:,:,:).^2));
ref_FDTD=squeeze(abs(field_ref_FDTD_scalar(:,:,:).^2));
ang_FDTD=squeeze(angle(field_trans_FDTD_scalar(:,:,:)./input_field_no_zero(:,:,:)/field_trans_FDTD_scalar(1,1,1)));

figure('Name','Transmission (Amp): CBS / FDTD');imagesc(cat(2,amp_CBS,amp_FDTD)); colormap gray;
figure('Name','Transmission (Phase): CBS / FDTD');imagesc(cat(2,ang_CBS,ang_FDTD)); colormap jet;
figure('Name','Reflection (Amp): CBS / FDTD');imagesc(cat(2,ref_CBS,ref_FDTD)); colormap gray;

intensity_CBS=sum(abs(field_CBS(round(end/2),round(end/2),:,:,1)).^2,4);
intensity_FDTD=sum(abs(field_FDTD(round(end/2),round(end/2),:,:,1)).^2,4);
intensity_Mie=sum(abs(field_3D_Mie(round(end/2),round(end/2),:,:,1)).^2,4);

figure('Name','Z-axis intensity profile througth the center of a bead'); hold on;
plot(squeeze(intensity_CBS),'.');
plot(squeeze(intensity_FDTD),'--');
plot(squeeze(intensity_Mie),'k');
legend('CBS', 'FDTD', 'Mie scattering'), title('Axial intensity')
%refractive index
MSE_CBS = mean(abs(field_CBS(:,:,:,:,1)-field_3D_Mie).^2, 'all');
MSE_FDTD = mean(abs(field_FDTD(:,:,:,:,1)-field_3D_Mie).^2, 'all');

figure('Name','Intensity: CBS / FDTD / Mie scattering');
orthosliceViewer(abs(cat(2,field_CBS(:,:,:,1,1),field_FDTD(:,:,:,1,1),field_3D_Mie(:,:,:,1,1))).^2);
colormap parula;
%H field
H_intensity_CBS=sum(abs(Hfield_CBS(:,:,:,:,1)).^2,4);
H_intensity_FDTD=sum(abs(Hfield_FDTD(:,:,:,:,1)).^2,4);
figure('Name','H field Intensity: CBS / FDTD');
orthosliceViewer(cat(2,H_intensity_CBS, H_intensity_FDTD));
colormap parula;
%% H field along z axis
H_intensity_diff=abs(abs(Hfield_CBS)-abs(Hfield_FDTD))./abs(Hfield_CBS);
for axis = 1:3
    disp(num2str(median(H_intensity_diff(:,:,:,axis),'all')))
end
orthosliceViewer(cat(2,H_intensity_diff(:,:,:,1),H_intensity_diff(:,:,:,2),H_intensity_diff(:,:,:,3)));
colormap parula;