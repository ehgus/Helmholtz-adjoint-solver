clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% User-defined optical parameter
use_GPU = true; % accelerator option
NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.025;  % unit: micron
diameter = 3;       % unit: micron
focal_length = 1; % unit: micron
z_padding = 0.5;    % padding along z direction
substrate_type = 'SU8';  % substrate type: 'SU8' or 'air'

% Refractive index profile
database = RefractiveIndexDB();
if strcmp(substrate_type,'SU8')
    substrate = database.material("other","resists","Microchem SU-8 2000");
    plate_thickness = 0.15;
elseif strcmp(substrate_type, 'air')
    substrate = @(x)1;
    plate_thickness = 0.35;
else
    error(['The substrate "' substrate_type '" is not supported'])
end
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
source_params.polarization = [1 0 0];
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;

current_source = PlaneSource(source_params);

%% Forward solver
params_CBS=params;
params_CBS.use_GPU=use_GPU;
params_CBS.boundary_thickness = [0 0 4];
params_CBS.field_attenuation = [0 0 4];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.RI_bg = double(sqrt((minRI^2+maxRI^2)/2));

forward_solver=ConvergentBornSolver(params_CBS);

% Configuration for bulk material
display_RI_Efield(forward_solver,RImap,current_source,'before optimization')

%% Adjoint method
x_pixel_coord = transpose((1:size(RImap,1))-diameter_pixel/2-1/2);
y_pixel_coord = (1:size(RImap,2))-diameter_pixel/2-1/2;
optim_region_xy = x_pixel_coord.^2 + y_pixel_coord.^2 < diameter_pixel^2/4;
optim_region = and(real(RImap) > 2, optim_region_xy);
regularizer_sequence = { ...
    AvgRegularizer('z'), ...
    CyclicConv2Regularizer(CyclicConv2Regularizer.conic(0.05/resolution),'z',@(step) step > 95), ...
    BinaryRegularizer(RI_list(2), RI_list(1), 1.5, 0.5, @(step) max(0,ceil((step-25)/25))) ...
};
grad_weight = 0.5;

% Adjoint solver
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Intensity";
adjoint_params.max_iter = 100;
adjoint_params.optimizer = FistaOptim(optim_region, regularizer_sequence, grad_weight);
adjoint_params.verbose = true;
adjoint_solver = AdjointSolver(adjoint_params);

% Adjoint design paramters
focal_spot_radius_pixel = 0.63 * wavelength * focal_length/diameter/resolution;
intensity_weight = phantom_bead([diameter_pixel, diameter_pixel, 2*round(z_padding/resolution)], [0 1], focal_spot_radius_pixel);
intensity_weight = padarray(intensity_weight,[0 0 sum(thickness_pixel)-size(intensity_weight,3)], 0,'pre');
options.intensity_weight = intensity_weight;

% Execute the optimization code
RI_optimized=adjoint_solver.solve(current_source,RImap,options);

%% Visualization
display_RI_Efield(forward_solver,RI_optimized,current_source,'after optimization')

%% optional: save RI configuration
filename = sprintf('optimized lens on %s Diameter-%.2fum F-%.2fum.mat',substrate_type,diameter,focal_length);
save_RI(filename, RI_optimized, params.resolution, params.wavelength);