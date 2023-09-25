clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% User-defined optical parameter
use_GPU = true; % accelerator option
NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.05;  % unit: micron
diameter = 10;       % unit: micron
focal_length = 3; % unit: micron
z_padding = 3;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'

% Refractive index profile
database = RefractiveIndexDB();
substrate = database.material("other","resists","Microchem SU-8 2000");
plate_thickness = 0.15;
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 substrate});
thickness_pixel = round([wavelength plate_thickness (focal_length + z_padding)]/resolution);
diameter_pixel = ceil(diameter/resolution);
RImap = phantom_plate([diameter_pixel diameter_pixel sum(thickness_pixel)], RI_list, thickness_pixel);

% Optical parameters
params.NA=NA;                   % Numerical aperture
params.wavelength=wavelength;   % unit: micron
[minRI, maxRI] = bounds(RI_list);
params.RI_bg=minRI;            % Background RI
params.resolution=ones(1,3) * resolution;         % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.size=size(RImap);        % 3D volume grid

% Incident field
source_params = params;
source_params.polarization = [1 -1i 0]/sqrt(2);
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;

current_source = PlaneSource(source_params);

%% Forward solver
params_CBS=params;
params_CBS.use_GPU=use_GPU;
params_CBS.boundary_thickness = [0 0 3];
params_CBS.field_attenuation = [0 0 3];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

forward_solver=ConvergentBornSolver(params_CBS);

% Configuration for bulk material
display_RI_Efield(forward_solver,RImap,current_source,'before optimization')

%% Adjoint method
optim_region = real(RImap) > 2;
regularizer_sequence = { ...
    AvgRegularizer('z'), ...
    CyclicConv2Regularizer(CyclicConv2Regularizer.conic(0.1/resolution),'z'), ...
    BinaryRegularizer2(RI_list(1), RI_list(2), 0.1, 0.1, @(step) max(ceil((step-40)/10), 0)), ...
    BinaryRegularizer(RI_list(1), RI_list(2), 0.1, 0.1, @(step) max(ceil((step-40)/10), 0)), ...
    MinimumLengthRegularizer(RI_list(1),RI_list(2),1,10,[0.1 0.9],@(step) ceil((step-50)/5) > 0) ...
};
grad_weight = 0.5;

% Adjoint solver
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 70;
adjoint_params.optimizer = FistaOptim(optim_region, regularizer_sequence, grad_weight);
adjoint_params.verbose = true;

adjoint_solver = AdjointSolver(adjoint_params);

% Adjoint design paramters
radius_pixel = round(0.1/resolution);
one_turn_length = round(2.5/resolution);
distance = round(0.65/resolution);
num_helix = 1;
intensity_weight = phantom_multi_helix([diameter_pixel, diameter_pixel, round(2*z_padding/resolution)], [0 1], radius_pixel, one_turn_length, distance, num_helix);
intensity_weight = intensity_weight .* reshape(exp(linspace(0,-1,size(intensity_weight,3))),1,1,[]);
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
options.intensity_weight = intensity_weight;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(current_source,RImap,options);

%% Visualization
forward_solver.set_RI(RI_optimized);
E_field = forward_solver.solve(current_source);
E_intensity = sum(abs(E_field),4);
viewerContinuous = viewer3d(BackgroundColor="white",BackgroundGradient="off",CameraZoom=2);
hVolumeContinuous = volshow(real(RI_optimized), OverlayData=E_intensity, Parent= viewerContinuous, OverlayAlphamap = linspace(0,0.2,256),...
    OverlayRenderingStyle = "GradientOverlay", RenderingStyle = "GradientOpacity", OverlayColormap=parula);

num_frames = 12;
dist = sqrt(sum(size(E_intensity).^2));
center = size(E_intensity)/2;
viewerContinuous.CameraTarget = center;
filename = "single_helical_structre.gif";
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

%% Optional: save RI configuration
filename = sprintf('optimized helical lens on %s Diameter-%.2fum F-%.2fum PSFlength-%.fum num-helix-%d.mat',substrate_type,diameter,focal_length,2*z_padding,num_helix);
save_RI(filename, RI_optimized, params.resolution, params.wavelength);