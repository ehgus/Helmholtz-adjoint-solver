clc, clear;close all
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));%% set the simulation parameters
%% 1 optical parameters
params.NA=1.2;
params.RI_bg=1.336;
params.wavelength=0.532;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=true;false;
params.use_abbe_sine=true;
params.size=[161 161 81];
%2 illumination parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
%3 phantom generation parameter
phantom_params=PHANTOM.get_default_parameters();
phantom_params.name='bead';%'RBC';
RI_sp=1.4609;
phantom_params.outer_size=params.size;
phantom_params.inner_size=round(ones(1,3) * 5 ./ params.resolution);
phantom_params.rotation_angles = [0 0 0];

%4 forward solver parameters
forward_MULTI_test1_params=params;
forward_MULTI_test1_params.use_GPU=true;
forward_MULTI_test1_params.return_3D=true;
forward_MULTI_test1_params.return_reflection=true;
forward_MULTI_test1_params.boundary_thickness=[1 1 2];
%forward_MULTI_test1_params.verbose=true;
forward_MULTI_test1_params.iterations_number=-1;

%4 forward solver parameters
forward_MULTI_test2_params=params;
forward_MULTI_test2_params.use_GPU=true;
forward_MULTI_test2_params.use_cuda=false;
forward_MULTI_test2_params.return_3D=true;
forward_MULTI_test2_params.return_reflection=true;
%forward_MULTI_test2_params.verbose=true;
forward_MULTI_test2_params.boundary_thickness=38;
forward_MULTI_test2_params.iterations_number=-1;

% forward solver - MIE
forward_params_Mie=params;
forward_params_Mie.use_GPU=true;
forward_params_Mie.truncated = true;
forward_params_Mie.verbose = true;
forward_params_Mie.boundary_thickness = 0;
forward_params_Mie.return_3D = true;
forward_params_Mie.lmax = 80;
forward_params_Mie.n_s = RI_sp;
forward_params_Mie.radius = 2.5;
forward_params_Mie.divide_section = 10;

%% create phantom and solve the forward problem
% make the phantom
RI=PHANTOM.get(phantom_params);
RI=params.RI_bg+RI.*(RI_sp-params.RI_bg);

%create the incident field
field_generator=FIELD_GENERATOR(field_generator_params);
input_field=field_generator.get_fields();
%compute the forward field
forward_MULTI_test1_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_MULTI_test1_params);
forward_MULTI_test1_solver.set_RI(RI);
tic;
[field_trans_multi_test1,field_ref_multi_test1,field_3D_multi_test1]=forward_MULTI_test1_solver.solve(input_field);
toc;

forward_MULTI_test2_solver=FORWARD_SOLVER_FDTD(forward_MULTI_test2_params);
forward_MULTI_test2_solver.set_RI(RI);
tic;
[field_trans_multi_test2,field_ref_multi_test2,field_3D_multi_test2]=forward_MULTI_test2_solver.solve(input_field);
toc;

%compute the forward field - Mie
if isfile('Mie_field.mat')
    load('Mie_field.mat')
else
    forward_solver_Mie=FORWARD_SOLVER_MIE(forward_params_Mie);
    forward_solver_Mie.set_RI(RI);
    tic;
    [field_trans_Mie,field_ref_Mie,field_3D_Mie]=forward_solver_Mie.solve(input_field);
    toc;
    imagesc(abs(field_trans_Mie(:,:,1)))
    save('Mie_field.mat', 'field_trans_Mie','field_ref_Mie','field_3D_Mie','-v7.3')
end

%% Draw results
field_3D_multi_test1 = field_3D_multi_test1/mean(field_3D_multi_test1,'all');
field_3D_multi_test2 = field_3D_multi_test2/mean(field_3D_multi_test2,'all');
field_3D_Mie = field_3D_Mie/mean(field_3D_Mie,'all');

[~,field_trans_multi_test1_scalar]=vector2scalarfield(input_field,field_trans_multi_test1);
[~,field_trans_multi_test2_scalar]=vector2scalarfield(input_field,field_trans_multi_test2);
[~,field_ref_multi_test1_scalar]=vector2scalarfield(input_field,field_ref_multi_test1);
[input_field_scalar,field_ref_multi_test2_scalar]=vector2scalarfield(input_field,field_ref_multi_test2);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));

disp_amp_multi_test1=squeeze(abs(field_trans_multi_test1_scalar(:,:,:)));
disp_ref_multi_test1=squeeze(abs(field_ref_multi_test1_scalar(:,:,:)));
disp_ang_multi_test1=squeeze(angle(field_trans_multi_test1_scalar(:,:,:)./input_field_no_zero(:,:,:)));

disp_amp_multi_test2=squeeze(abs(field_trans_multi_test2_scalar(:,:,:)));
disp_ref_multi_test2=squeeze(abs(field_ref_multi_test2_scalar(:,:,:)));
disp_ang_multi_test2=squeeze(angle(field_trans_multi_test2_scalar(:,:,:)./input_field_no_zero(:,:,:)));


figure;imagesc(cat(2,disp_amp_multi_test1,disp_amp_multi_test2)); colormap gray;
figure;imagesc(cat(2,disp_ref_multi_test1,disp_ref_multi_test2)); colormap gray;
figure;imagesc(cat(2,disp_ang_multi_test1,disp_ang_multi_test2)); colormap jet;

vert_intensity_1=sum(abs(field_3D_multi_test1(round(end/2),round(end/2),:,:,1)).^2,4);
vert_intensity_2=sum(abs(field_3D_multi_test2(round(end/2),round(end/2),:,:,1)).^2,4);
vert_intensity_3=sum(abs(field_3D_Mie(round(end/2),round(end/2),:,:,1)).^2,4);

figure; 
plot(squeeze(vert_intensity_1),'.'); hold on;
plot(squeeze(vert_intensity_2),'--'); hold on;
plot(squeeze(vert_intensity_3),'k'); hold on;
legend('Acyclic + analytic', 'Cyclic + Sheet source', 'Mie theory'), title('Axial intensity')
%refractive index
MSE_test1 = mean(abs(field_3D_multi_test1(:,:,:,:,1)-field_3D_Mie).^2, 'all');
MSE_test2 = mean(abs(field_3D_multi_test2(:,:,:,:,1)-field_3D_Mie).^2, 'all');

figure('Name','CBS / FDTD / Mie scattering');
orthosliceViewer(abs(cat(2,field_3D_multi_test1(:,:,:,1),field_3D_multi_test2(:,:,:,1),field_3D_Mie(:,:,:,1))));
colormap parula;