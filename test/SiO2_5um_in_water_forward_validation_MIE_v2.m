clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));%% set the simulation parameters
%% 1 optical parameters
radius = 1.5;
params.NA=1.2;
params.RI_bg=1.336;
RI_sp=1.4609;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/8/params.NA;
params.vector_simulation=true;
params.use_abbe_sine=true;
params.size= [161 161 81];
%params.size=[501 501 81];
%params.size= [171 171 91];

% forward solver parameters - CBS
forward_MULTI_test1_params=params;
forward_MULTI_test1_params.use_GPU=false;
forward_MULTI_test1_params.return_3D=true;
forward_MULTI_test1_params.return_reflection=true;
forward_MULTI_test1_params.boundary_thickness=[0 0 6];%[1 1 2];
forward_MULTI_test1_params.verbose=false;
forward_MULTI_test1_params.iterations_number=-1;

% forward solver parameters - FDTD
forward_MULTI_test2_params=params;
forward_MULTI_test2_params.use_GPU=false;
forward_MULTI_test2_params.use_cuda=false;
forward_MULTI_test2_params.return_3D=true;
forward_MULTI_test2_params.return_reflection=true;
forward_MULTI_test2_params.verbose=false;
forward_MULTI_test2_params.boundary_thickness=[0 0 38];
forward_MULTI_test2_params.iterations_number=-1;
forward_MULTI_test2_params.is_plane_wave = true;
forward_MULTI_test2_params.fdtd_temp_dir = fullfile(dirname,'test/FDTD_TEMP');

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
forward_MULTI_test1_solver=ConvergentBornSolver(forward_MULTI_test1_params);
forward_MULTI_test1_solver.set_RI(RI);
tic;
[field_trans_multi_test1,field_ref_multi_test1,field_3D_multi_test1]=forward_MULTI_test1_solver.solve(input_field);
toc;

forward_MULTI_test2_solver=FDTDsolver(forward_MULTI_test2_params);
forward_MULTI_test2_solver.set_RI(RI);
tic;
[field_trans_multi_test2,field_ref_multi_test2,field_3D_multi_test2]=forward_MULTI_test2_solver.solve(input_field);
toc;

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
% field_3D_multi_test1 = field_3D_multi_test1/median(field_3D_multi_test1,'all');
% field_3D_multi_test2 = field_3D_multi_test2/median(field_3D_multi_test2,'all');
% field_3D_Mie = field_3D_Mie/median(field_3D_Mie,'all');

[~,field_trans_multi_test1_scalar]=vector2scalarfield(input_field,field_trans_multi_test1);
[~,field_trans_multi_test2_scalar]=vector2scalarfield(input_field,field_trans_multi_test2);
[~,field_ref_multi_test1_scalar]=vector2scalarfield(input_field,field_ref_multi_test1);
[input_field_scalar,field_ref_multi_test2_scalar]=vector2scalarfield(input_field,field_ref_multi_test2);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));

disp_amp_multi_test1=squeeze(abs(field_trans_multi_test1_scalar(:,:,:).^2));
disp_ref_multi_test1=squeeze(abs(field_ref_multi_test1_scalar(:,:,:).^2));
disp_ang_multi_test1=squeeze(angle(field_trans_multi_test1_scalar(:,:,:)./input_field_no_zero(:,:,:)/field_trans_multi_test1_scalar(1,1,1)));

disp_amp_multi_test2=squeeze(abs(field_trans_multi_test2_scalar(:,:,:).^2));
disp_ref_multi_test2=squeeze(abs(field_ref_multi_test2_scalar(:,:,:).^2));
disp_ang_multi_test2=squeeze(angle(field_trans_multi_test2_scalar(:,:,:)./input_field_no_zero(:,:,:)/field_trans_multi_test2_scalar(1,1,1)));


figure('Name','Transmission (Amp): CBS / FDTD');imagesc(cat(2,disp_amp_multi_test1,disp_amp_multi_test2)); colormap gray;
figure('Name','Transmission (Phase): CBS / FDTD');imagesc(cat(2,disp_ang_multi_test1,disp_ang_multi_test2)); colormap jet;
figure('Name','Reflection (Amp): CBS / FDTD');imagesc(cat(2,disp_ref_multi_test1,disp_ref_multi_test2)); colormap gray;

vert_intensity_1=sum(abs(field_3D_multi_test1(round(end/2),round(end/2),:,:,1)).^2,4);
vert_intensity_2=sum(abs(field_3D_multi_test2(round(end/2),round(end/2),:,:,1)).^2,4);
vert_intensity_3=sum(abs(field_3D_Mie(round(end/2),round(end/2),:,:,1)).^2,4);

figure('Name','Z-axis intensity profile througth the center of a bead'); hold on;
plot(squeeze(vert_intensity_1),'.');
plot(squeeze(vert_intensity_2),'--');
plot(squeeze(vert_intensity_3),'k');
legend('CBS', 'FDTD', 'Mie scattering'), title('Axial intensity')
%refractive index
MSE_test1 = mean(abs(field_3D_multi_test1(:,:,:,:,1)-field_3D_Mie).^2, 'all');
MSE_test2 = mean(abs(field_3D_multi_test2(:,:,:,:,1)-field_3D_Mie).^2, 'all');

figure('Name','Intensity: CBS / FDTD / Mie scattering');
orthosliceViewer(abs(cat(2,field_3D_multi_test1(:,:,:,1,1),field_3D_multi_test2(:,:,:,1,1),field_3D_Mie(:,:,:,1,1))).^2);
colormap parula;