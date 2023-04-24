clc;clear;close all;

% 3D: (pol, prop) = (x,z)
% 2D: (pol, prop) = (-z,-y)
% x -> y
% y -> -z
% z -> -x
% E_field_3D = reshape(E_field.E,length(E_field.x),length(E_field.y),length(E_field.z),3);
% E_field_3D =  E_field_3D(:,:,:,[3 1 2]); E_field_3D(:,:,:,[1,3]) = -E_field_3D(:,:,:,[1,3]);
% E_field_3D = permute(E_field_3D, [3 1 2 4]);E_field_3D = flip(E_field_3D, 1);E_field_3D = flip(E_field_3D, 3);
% E_field_3D = E_field_3D(:,:,end-106:end-1,:); ## do the same thing to
% H_field_3D
% save('grating_pattern_2D_FDTD.mat','E_field_3D','H_field_3D')
% crop the results?
EMfield = struct( ...
'FDTD_2D_field', load('grating_pattern_2D_FDTD.mat'), ...
'FDTD_3D_field', load('grating_pattern_FDTDsolver_oversample_1.mat'), ...
'CBS_field', load('grating_pattern_ConvergentBornSolver_oversample_1.mat') ...
);

target_transmission = [0 0 0.17 0 0.64 0 0.17];
measure_slice = 29;

EMfield_name = fieldnames(EMfield);
for i = 1:length(EMfield_name)
    if i == 1
        x_center = 1;
    else
        x_center = 51;
    end
    ref_angle = EMfield.(EMfield_name{i}).E_field_3D(x_center,51,end-measure_slice,1);
    ref_angle = ref_angle/abs(ref_angle)*exp(-1i*1.40291);
    EMfield.(EMfield_name{i}).E_field_3D = EMfield.(EMfield_name{i}).E_field_3D./ref_angle;
    EMfield.(EMfield_name{i}).H_field_3D = EMfield.(EMfield_name{i}).H_field_3D./ref_angle;
end

%% Reference field (3D)
% 0. forward solver
forward_params.NA=1; % Numerical aperture
forward_params.wavelength=0.355; % [um]
forward_params.RI_bg=1.4; % Background RI
forward_params.resolution=[0.01 0.01 0.01]; % 3D Voxel size [um]
forward_params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
forward_params.vector_simulation=true; % True/false: dyadic/scalar Green's function
forward_params.size=[101 101 106]; % 3D volume grid
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];
forward_params.RI_bg = 1.4;

forward_solver=ConvergentBornSolver(forward_params);

% 1. Reference field

ref_E_field = cell(1,length(target_transmission));
ref_H_field = cell(1,length(target_transmission));
normal_transmission = cell(1,length(target_transmission));

impedance = 377/forward_solver.RI_bg;
Nsize = forward_solver.size + 2*forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(ref_E_field)
    % E field
    illum_order = i - 3;
    sin_theta = (illum_order-1)*forward_params.wavelength/(forward_params.size(2)*forward_params.resolution(2)*forward_solver.RI_bg);
    cos_theta = sqrt(1-sin_theta^2);
    if illum_order < 1
        illum_order = illum_order + Nsize(1);
    end
    incident_field = zeros(Nsize([1 2 4]));
    incident_field(1,illum_order,1) = prod(Nsize(1:2));
    incident_field = ifft2(incident_field);
    incident_field = forward_solver.padd_field2conv(incident_field);
    incident_field = fft2(incident_field);
    incident_field = reshape(incident_field, [size(incident_field,1),size(incident_field,2),1,size(incident_field,3)]).*forward_solver.refocusing_util;
    incident_field = ifft2(incident_field);
    ref_E_field{i} = forward_solver.crop_conv2RI(incident_field);
    % angle correction
    center_angle = ref_E_field{i}(51,51,forward_solver.ROI(6)-measure_slice,1);
    ref_E_field{i} = ref_E_field{i}/center_angle;
    % H field
    incident_field_H = zeros(Nsize,'like',ref_E_field{i});
    incident_field_H(:,:,:,2) = ref_E_field{i}(:,:,:,1) ;%* cos_theta;
    incident_field_H(:,:,:,3) = ref_E_field{i}(:,:,:,1) ;%* (-sin_theta);
    incident_field_H = incident_field_H/impedance;
    ref_H_field{i} = incident_field_H;
    
    % normal transmission
    normal_S = 2 * real(poynting_vector(ref_E_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:), ...
                                        ref_H_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)));
    normal_transmission{i} = abs(sum(normal_S(:,:,end-measure_slice,3),1:2));
    %disp(normal_transmission{i}(1))
end
%% display field (check fields are well structued)

% E field (amp)
figure('Name','E field (amp)');
for i = 1:length(EMfield_name)
    subplot(1,3,i)
    imagesc(squeeze(abs(EMfield.(EMfield_name{i}).E_field_3D(1,:,:,1)))');
    title(strrep(EMfield_name{i},'_',' '));
end

% H field (amp)
figure('Name','H field (amp)');
for i = 1:length(EMfield_name)
    subplot(1,3,i)
    imagesc(squeeze(abs(EMfield.(EMfield_name{i}).H_field_3D(1,:,:,3)))');
    title(strrep(EMfield_name{i},'_',' '));
end

% E field (phase)
figure('Name','E field (phase)');
for i = 1:length(EMfield_name)
    subplot(1,3,i)
    imagesc(squeeze(angle(EMfield.(EMfield_name{i}).E_field_3D(1,:,:,1)))');
    title(strrep(EMfield_name{i},'_',' '));
end

% H field (phase)
figure('Name','H field (phase)');
for i = 1:length(EMfield_name)
    subplot(1,3,i)
    imagesc(squeeze(angle(EMfield.(EMfield_name{i}).H_field_3D(1,:,:,3)))');
    title(strrep(EMfield_name{i},'_',' '));
end

%% 1.1 reproduce the python's result
FDTD_2D_field_slice = load('grating_pattern_2D_FDTD_slice.mat');
% cal_overlap_int

n = 1.40;
impedance = 377/n;

num_order = length(target_transmission);
coord_length = size(FDTD_2D_field_slice.E_field_2D,2);
phase_factors = exp(1i*reshape(-3:3,[],1).*reshape(linspace(-pi,pi,coord_length),1,[]));
Em_plus = zeros(num_order,coord_length,3);
Hm_plus = zeros(num_order,coord_length,3);

% method 1
for i = 1:num_order
    Em_plus(i,:,:) = reshape(ref_E_field{i}(1,:,forward_solver.ROI(6)-measure_slice,[2 3 1]),1,[],3);
    Hm_plus(i,:,:) = reshape(ref_H_field{i}(1,:,forward_solver.ROI(6)-measure_slice,[2 3 1]),1,[],3);
    Hm_plus(i,:,1) = -Hm_plus(i,:,1);
end
% % method 2
% Em_plus(:,:,3) = phase_factors;
% Hm_plus(:,:,1) = - phase_factors /impedance;

Em_minus = conj(Em_plus);
Hm_minus = conj(Hm_plus);

normal_integrand = cross(Em_plus, Hm_minus,3) + cross(Em_minus, Hm_plus,3);
normal_integral = abs(sum(normal_integrand(:,:,2),2));

integrand = cross(repmat(FDTD_2D_field_slice.E_field_2D, [num_order 1 1]), Hm_minus,3) + cross(Em_minus, repmat(FDTD_2D_field_slice.H_field_2D, [num_order 1 1]),3);
integral = sum(integrand(:,:,2), 2);

rst = -integral./normal_integral;
disp(angle(rst/rst(end))*180/pi)
%    method 1  method 2
%    -78.9422  -78.5969
%    -45.4444  -45.4905
%    116.6384  117.9341
%    -35.2242  -33.9769
%    -81.9462  -80.8807
%    -44.3525  -16.6663
%           0         0
disp(abs(rst).^2)
%    method 1  method 2
%     0.0030    0.0029
%     0.0015    0.0017
%     0.1580    0.1548
%     0.0003    0.0003
%     0.6209    0.6177
%     0.0011    0.0010
%     0.1591    0.1588
%% 1.2-a extend the following result in 3D field

E = EMfield.CBS_field.E_field_3D(1,:,end-measure_slice,:);
H = EMfield.CBS_field.H_field_3D(1,:,end-measure_slice,:);

% normal_integral = zeros(length(ref_E_field),1);
% integral = zeros(length(ref_E_field),1);
% figure(101);hold on; plot(E(:,:,:,1));plot(FDTD_2D_field_slice.E_field_2D(:,:,3));
% figure(202);hold on; plot(H(:,:,:,2));plot(-FDTD_2D_field_slice.H_field_2D(:,:,1));



for i = 1:length(ref_E_field)

%     % method 1
%     Em_plus = ref_E_field{i}(1,:,forward_solver.ROI(6)-measure_slice,:);
%     Hm_plus = ref_H_field{i}(1,:,forward_solver.ROI(6)-measure_slice,:);
    % method 2
    phase_factors = exp(1i*(i-4)*reshape(linspace(-pi,pi,coord_length),1,[]));
    Em_plus = zeros(1,coord_length,1,3);
    Hm_plus = zeros(1,coord_length,1,3);
    Em_plus(:,:,:,1) = phase_factors;
    Hm_plus(:,:,:,2) = phase_factors /impedance;

    Em_minus = conj(Em_plus);
    Hm_minus = conj(Hm_plus);

    normal_integrand = cross(Em_plus, Hm_minus,4) + cross(Em_minus, Hm_plus,4);
    normal_integral(i) = abs(sum(normal_integrand(:,:,:,3),'all'));

    integrand = cross(E, Hm_minus,4) + cross(Em_minus, H,4);

    integral(i) = sum(integrand(:,:,:,3),'all');
end



rst = -integral./normal_integral;
disp(angle(rst)*180/pi)
%  method1 (2D FDTD)  method 2 (2D FDTD) method1 (3D FDTD)  method 2 (3D FDTD)    method1 (CBS)  method2 (CBS) 
%       -129.3355       -130.2376          -30.0167              -30.5754           -118.9382       -115.5522
%        -95.8378        -97.1311         -178.6327             -174.2086            121.9908        129.4714
%         66.2450         66.2935           65.0243               65.0700             57.7832         57.7092
%        -85.6175        -85.6176          120.8238              120.8238            157.4695        157.4695
%       -132.3395       -132.5213         -133.0346             -133.1514           -137.0580       -137.1562
%        -94.7458        -68.3069         -107.2019              -87.4837            -42.9643        -21.5187
%        -50.3934        -51.6407          -53.2373              -54.3970            -47.4344        -48.7811
disp(angle(rst/rst(end))*180/pi)
%  method1 (2D FDTD)  method 2 (2D FDTD) method1 (3D FDTD)  method 2 (3D FDTD)    method1 (CBS)  method2 (CBS) 
%        -78.9422         -78.5969           23.2206              23.8215            -71.5039       -66.7711
%        -45.4444         -45.4905         -125.3954            -119.8117            169.4252       178.2525
%        116.6384         117.9341          118.2617             119.4670            105.2175       106.4903
%        -35.2242         -33.9769          174.0612             175.2208           -155.0961      -153.7494
%        -81.9462         -80.8807          -79.7972             -78.7545            -89.6237       -88.3752
%        -44.3525         -16.6663          -53.9646             -33.0867              4.4701        27.2624
%               0                0                 0                    0                   0     	       0
disp(abs(rst).^2)
%    method1 (2D FDTD)  method 2 (2D FDTD) method1 (3D FDTD)  method 2 (3D FDTD)     method1 (CBS)  method2 (CBS)
%        0.0030              0.0029              0.0017              0.0018              0.0019          0.0019
%        0.0015              0.0017              0.0005              0.0005              0.0006          0.0006
%        0.1580              0.1548             `0.1326`            `0.1295`             0.1411          0.1385
%        0.0003              0.0003              0.0035              0.0035              0.0065          0.0065
%        0.6209              0.6177             `0.5840`            `0.5815`             0.5920          0.5887
%        0.0011              0.0010              0.0016              0.0013              0.0005          0.0011
%        0.1591              0.1588             `0.1615`            `0.1624`             0.1442          0.1439

% ## interpretation
% Considering analysis of field from 2D FDTD as reference, the result is
% quite consistent.
% The absolute value is slightly degraded.
%% 2.1 matlab-based mode decomposition
% 0. forward solver
forward_params.NA=1; % Numerical aperture
forward_params.wavelength=0.355; % [um]
forward_params.RI_bg=1.4; % Background RI
forward_params.resolution=[0.01 0.01 0.01]; % 3D Voxel size [um]
forward_params.use_abbe_sine=false; % Abbe sine condition according to demagnification condition
forward_params.vector_simulation=true; % True/false: dyadic/scalar Green's function
forward_params.size=[101 101 106]; % 3D volume grid
forward_params.use_GPU=true;
forward_params.return_3D = true;
forward_params.boundary_thickness = [0 0 4];
forward_params.RI_bg = 1.4;

forward_solver=ConvergentBornSolver(forward_params);

% 2. field decomposition
relative_transmission = zeros(1,length(target_transmission));
for j = 1:length(EMfield_name)
    E_field_3D = EMfield.(EMfield_name{j}).E_field_3D;
    H_field_3D = EMfield.(EMfield_name{j}).H_field_3D;
    
    for i = 1:length(target_transmission)
        eigen_S = poynting_vector(E_field_3D(1,:,:,:), ref_H_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)) ...
                + poynting_vector(conj(ref_E_field{i}(1,forward_solver.ROI(3):forward_solver.ROI(4),forward_solver.ROI(5):forward_solver.ROI(6),:)), conj(H_field_3D(1,:,:,:)));
        relative_transmission(i) = mean(sum(eigen_S(:,:,end-measure_slice,3),1:2)./normal_transmission{i},'all');
    end
    adjoint_source_weight = abs(relative_transmission).^2;
    disp(strrep(EMfield_name{j},'_',' '));
    %t_conj = conj(relative_transmission);
    disp(angle(-relative_transmission) * (180 / pi))
    disp(angle(relative_transmission/relative_transmission(end)) * (180 / pi))
    disp(adjoint_source_weight)
end
% FDTD 2D field
%  -129.3355  -95.8378   66.2450  -85.6175 -132.3395  -94.7458  -50.3934
% 
%   -78.9422  -45.4444  116.6384  -35.2242  -81.9462  -44.3525         0
% 
%     0.0030    0.0015    0.1580    0.0003    0.6209    0.0011    0.1591
% 
% FDTD 3D field
%   -30.0167 -178.6327   65.0243  120.8238 -133.0346 -107.2019  -53.2373
% 
%    23.2206 -125.3954  118.2617  174.0612  -79.7972  -53.9646         0
% 
%     0.0017    0.0005    0.1326    0.0035    0.5840    0.0016    0.1615
% 
% CBS field
%  -118.9382  121.9908   57.7832  157.4695 -137.0580  -42.9643  -47.4344
% 
%   -71.5039  169.4252  105.2175 -155.0961  -89.6237    4.4701         0
% 
%     0.0019    0.0006    0.1411    0.0065    0.5920    0.0005    0.1442

% ## interpretation
% The results is consistent with the previous results

%% 2.2 constructed field profile
slice_E_field = EMfield.(EMfield_name{1}).E_field_3D(1,:,end-measure_slice,1);
figure;
subplot(2,1,1)
plot(angle(slice_E_field))
subplot(2,1,2)
plot(abs(slice_E_field))

%% 2.3 incident field profile
slice_E_field = ref_E_field{5}(1,:,end-measure_slice,1);
figure;
subplot(2,1,1)
plot(angle(slice_E_field))
subplot(2,1,2)
plot(abs(slice_E_field))