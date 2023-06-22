clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% user-defined optical parameter
use_GPU = true; % accelerator option

NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.025;  % unit: micron
diameter = 10;       % unit: micron
focal_length = 0; % unit: micron
z_padding = 3;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'
verbose = false;

% refractive index profile
database = RefractiveIndexDB();
substrate = database.material("other","resists","Microchem SU-8 2000");
plate_thickness = 0.3;
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 substrate});
thickness_pixel = round([wavelength plate_thickness z_padding]/resolution);
diameter_pixel = ceil(diameter/resolution);
RImap = phantom_plate([diameter_pixel diameter_pixel sum(thickness_pixel)], RI_list, thickness_pixel);

%1 optical parameters
params.NA=NA;                   % Numerical aperture
params.wavelength=wavelength;   % unit: micron
[minRI, maxRI] = bounds(RI_list);
params.RI_bg=minRI;            % Background RI
params.resolution=ones(1,3) * resolution;         % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.vector_simulation=true;  % True/false: dyadic/scalar Green's function
params.size=size(RImap);        % 3D volume grid

%2 incident field parameters
field_generator_params=params;
field_generator_params.illumination_number=1;
field_generator_params.illumination_style='circle';%'circle';%'random';%'mesh'
% create the incident field
input_field_x = FieldGenerator.get_field(field_generator_params);
input_field_y = -1i .* flip(input_field_x, 3);
input_field = (input_field_x + input_field_y)./sqrt(2);

%% Forward solver

params_CBS=params;
params_CBS.use_GPU=use_GPU;
params_CBS.return_3D = true;
params_CBS.boundary_thickness = [0 0 4];
params_CBS.field_attenuation = [0 0 4];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(params_CBS);
forward_solver.set_RI(RImap);

% Configuration for bulk material
if verbose
    display_RI_Efield(forward_solver,RImap,input_field,'before optimization')
end
%% Adjoint method
ROI_change = real(RImap) > 2;
forward_solver.return_transmission = false;
forward_solver.return_reflection = false;
%Adjoint solver parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Intensity";
adjoint_params.ROI_change = ROI_change;
adjoint_params.step = 0.5;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 0.5;
adjoint_params.binarization_step = 40;
adjoint_params.spatial_diameter = 0.1;
adjoint_params.spatial_filter_range = [Inf Inf];
adjoint_params.nmin = RI_list(1);
adjoint_params.nmax = RI_list(2);
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [false false true];
tic;
adjoint_solver = AdjointSolver(adjoint_params);
toc;
radius_pixel = round(0.1/resolution);
one_turn_length = round(2/resolution);
distance = round(0.5/resolution);
num_helix = 2;
intensity_weight = phantom_multi_helix([diameter_pixel, diameter_pixel, round((z_padding)/2/resolution)], [-0.02 1], radius_pixel, one_turn_length, distance, num_helix);
%intensity_weight = padarray(intensity_weight,[0 0 round(z_padding/4/resolution)], 0, 'post');
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
filter_axis = exp(-(-2:2).^2);
blur_filter = reshape(filter_axis,[],1).*reshape(filter_axis,1,[]).*reshape(filter_axis,1,1,[]);
blur_filter = blur_filter./sum(blur_filter,'all');
intensity_weight = convn(intensity_weight, blur_filter, 'same');
options.intensity_weight = intensity_weight;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(input_field,RImap,options);

%% visualization
[~,~,E_field] = forward_solver.solve(input_field);
E_intensity = sum(abs(E_field),4);
viewerContinuous = viewer3d(BackgroundColor="white",BackgroundGradient="off",CameraZoom=2);
hVolumeContinuous = volshow(real(RI_optimized), OverlayData=E_intensity, Parent= viewerContinuous, OverlayAlphamap = linspace(0,0.2,256),...
    OverlayRenderingStyle = "GradientOverlay", RenderingStyle = "GradientOpacity");

num_frames = 12;
dist = sqrt(sum(size(E_intensity).^2));
center = size(E_intensity)/2;
viewerContinuous.CameraTarget = center;
filename = "helical_structre.gif";
for idx = 1:num_frames
    angle = 2*pi*idx/num_frames;
    viewerContinuous.CameraPosition = center + ([cos(angle) sin(angle) 0.1]*dist);
    I = getframe(viewerContinuous.Parent);
    [idxI, cm] = rgb2ind(I.cdata, 256);
    if idx == 1
        imwrite(idxI,cm,filename,"gif",Loopcount=inf,DelayTime=0.1)
    else
        imwrite(idxI,cm,filename,"gif",WriteMode="append",DelayTime=0.1)
    end
end
%% optional: save RI configuration
filename = sprintf('optimized helical lens on %s Diameter-%.2fum num-helix-%d.mat',substrate_type,diameter,num_helix);
save_RI(filename, RI_optimized, params.resolution, params.wavelength);