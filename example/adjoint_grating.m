clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% User-defined optical parameter
use_GPU = true; % accelerator option
NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.01; % unit: micron
mask_width = 0.15;
grid_size = [11 100 100];
target_transmission = [0 0 0.15 0 0.63 0 0.22];
verbose = false;

% Refractive index profile
RI_database = RefractiveIndexDB();
PDMS = RI_database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = RI_database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = RI_database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) func(wavelength), {PDMS TiO2 Microchem_SU8_2000});
thickness_pixel = [0.20 mask_width]/resolution;
RI = phantom_plate(grid_size, RI_list, thickness_pixel);

% Optical parameters
params.NA = NA;
params.wavelength = wavelength;
params.RI_bg=RI_list(1);        % Background RI - should be matched with mode decomposition area
params.resolution=ones(1,3) * resolution; % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.size=grid_size;          % 3D volume grid

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
params_CBS.boundary_thickness = [0 0 3];
params_CBS.field_attenuation = [0 0 3];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation_sharpness = 0.5;
params_CBS.verbose = false;
forward_solver=ConvergentBornSolver(params_CBS);

% Configuration for bulk material
display_RI_Efield(forward_solver,RI,current_source,'before optimization');

%% Adjoint solver
optim_region = zeros(size(RI),'logical');
optim_region(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = true;
regularizer_sequence = { ...
    BoundRegularizer(RI_list(1), RI_list(2)) ...
};
density_projection_sequence = { ...
    BoundRegularizer(0, 1), ...
    AvgRegularizer('xz') ...
};
grad_weight = 8e-4;

% Adjoint solver
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.optim_mode = "Transmission";
adjoint_params.optimizer = Optim(optim_region, regularizer_sequence, density_projection_sequence, grad_weight);
adjoint_params.max_iter = 100;
adjoint_params.verbose = true;
adjoint_params.sectioning_axis = "x";

adjoint_solver = AdjointSolver(adjoint_params);

% Adjoint design paramters
options = struct;
options.target_transmission = target_transmission;
options.surface_vector = zeros(adjoint_params.size(1),adjoint_params.size(2),adjoint_params.size(3),3);
options.surface_vector(:,:,end,:) = options.surface_vector(:,:,end,:) + reshape([0 0 1],1,1,1,3);
params_CBS.RI_bg = RI_list(3);
options.forward_solver = ConvergentBornSolver(params_CBS);

for i = 1:length(options.target_transmission)
    illum_order = i - 4;
    k_y = 2*pi*illum_order/(params_CBS.size(2)*params_CBS.resolution(2));
    adj_source_params = params_CBS;
    adj_source_params.polarization = [-1 0 0];
    adj_source_params.direction = 3;
    adj_source_params.grid_size = adj_source_params.size;
    adj_source_params.outcoming_wave = false;
    adj_source_params.horizontal_k_vector = [0 -k_y];
    adj_source_params.center_position = [1 1 adj_source_params.size(3)+1];

    adj_current_source = PlaneSource(adj_source_params);
    options.current_source(i) = adj_current_source;
end

% Execute the optimization code
RI_optimized=adjoint_solver.solve(current_source,RI,options);

%% Visualization
display_RI_Efield(forward_solver,RI_optimized,current_source,'after optimization')

%% Optional: save RI configuration
filename = 'CBS_optimized grating.mat';
save_RI(filename, RI_optimized(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))), params.resolution, params.wavelength);