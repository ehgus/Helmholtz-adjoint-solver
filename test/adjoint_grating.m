 clc, clear;
dirname = fileparts(fileparts(matlab.desktop.editor.getActiveFilename));
addpath(genpath(dirname));

%% Basic optical parameters

% user-defined optical parameter
use_GPU = true; % accelerator option

NA = 1;             % numerical aperature
wavelength = 0.355; % unit: micron
resolution = 0.01; % unit: micron
mask_width = 0.15;
grid_size = [11 100 100];
target_transmission = [0 0 0.15 0 0.63 0 0.22];
verbose = false;

% refractive index profile
database = RefractiveIndexDB();
PDMS = database.material("organic","(C2H6OSi)n - polydimethylsiloxane","Gupta");
TiO2 = database.material("main","TiO2","Siefke");
Microchem_SU8_2000 = database.material("other","resists","Microchem SU-8 2000");
RI_list = cellfun(@(func) real(func(wavelength)), {PDMS TiO2 Microchem_SU8_2000});
thickness_pixel = [0.20 mask_width]/resolution;
RI = phantom_plate(grid_size, RI_list, thickness_pixel);

%1 optical parameters
params.NA = NA;
params.wavelength = wavelength;
params.RI_bg=RI_list(1);        % Background RI - should be matched with mode decomposition area
params.resolution=ones(1,3) * resolution; % 3D Voxel size [um]
params.use_abbe_sine=false;     % Abbe sine condition according to demagnification condition
params.size=grid_size;          % 3D volume grid

%2 incident field parameters
source_params = params;
source_params.polarization = [1 0 0];
source_params.direction = 3;
source_params.horizontal_k_vector = [0 0];
source_params.center_position = [1 1 1];
source_params.grid_size = source_params.size;
current_source = PlaneSource(source_params);

%% forward solver

params_CBS=params;
params_CBS.use_GPU=use_GPU;

params_CBS.boundary_thickness = [0 0 5];
params_CBS.field_attenuation = [0 0 5];
params_CBS.field_attenuation_sharpness = 0.5;
params_CBS.potential_attenuation = [0 0 4];
params_CBS.potential_attenuation_sharpness = 0.5;

%compute the forward field using convergent Born series
forward_solver=ConvergentBornSolver(params_CBS);
display_RI_Efield(forward_solver,RI,current_source,'before optimization');
%% Adjoint solver
forward_solver.verbose = false;
ROI_change = zeros(size(RI),'logical');
ROI_change(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))) = true;
%Adjoint solver iteration parameters
adjoint_params=params;
adjoint_params.forward_solver = forward_solver;
adjoint_params.mode = "Transmission";
adjoint_params.ROI_change = ROI_change;
adjoint_params.step = 3;
adjoint_params.itter_max = 100;
adjoint_params.steepness = 2;
adjoint_params.binarization_step = 150;
adjoint_params.spatial_diameter = 0.1;
adjoint_params.spatial_filter_range = [Inf Inf];
adjoint_params.nmin = RI_list(1);
adjoint_params.nmax = RI_list(2);
adjoint_params.verbose = true;
adjoint_params.averaging_filter = [true false true];
%adjoint_params.optimizer = Optim;

adjoint_solver = AdjointSolver(adjoint_params);
%Adjoint field parameter
options = struct;
options.target_transmission = target_transmission;
options.surface_vector = zeros(adjoint_params.size(1),adjoint_params.size(2),adjoint_params.size(3),3);
options.surface_vector(:,:,end,:) = options.surface_vector(:,:,end,:) + reshape([0 0 1],1,1,1,3);
options.E_field = cell(1,length(options.target_transmission));
options.H_field = cell(1,length(options.target_transmission));
params_CBS.RI_bg = RI_list(3);
options.forward_solver = ConvergentBornSolver(params_CBS);
impedance = 377/options.forward_solver.RI_bg;
Nsize = options.forward_solver.size + 2*options.forward_solver.boundary_thickness_pixel;
Nsize(4) = 3;

for i = 1:length(options.E_field)
    illum_order = i - 4;
    k_y = 2*pi*illum_order/(params_CBS.size(2)*params_CBS.resolution(2));
    adj_source_params = params_CBS;
    adj_source_params.polarization = [-1 0 0];
    adj_source_params.direction = 3;
    adj_source_params.horizontal_k_vector = [0 k_y];
    adj_source_params.center_position = [1 1 1-options.forward_solver.boundary_thickness_pixel(3)];
    adj_source_params.grid_size = adj_source_params.size;
    adj_current_source = PlaneSource(adj_source_params);
    options.current_source{i} = adj_current_source;
    options.E_field{i} = adj_current_source.generate_Efield(repmat(options.forward_solver.boundary_thickness_pixel,2,1));
    options.H_field{i} = adj_current_source.generate_Hfield(repmat(options.forward_solver.boundary_thickness_pixel,2,1));
end

% Execute the optimization code
RI_optimized=adjoint_solver.solve(current_source,RI,options);
% Configuration for optimized metamaterial
display_RI_Efield(forward_solver,RI_optimized,current_source,'after optimization')
%% optional: save RI configuration
filename = 'CBS_optimized grating.mat';
save_RI(filename, RI_optimized(:,:,thickness_pixel(1)+1:sum(thickness_pixel(1:2))), params.resolution, params.wavelength);