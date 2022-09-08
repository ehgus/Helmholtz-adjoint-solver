classdef OPTICAL_SYSTEM < STRUCT_CLASS
    properties
        NA = 1.2;                       % numerical aperture
        wavelength=0.532;        % wavelength (um)
        RI_bg=1.336;             % media RI
        resolution=[0.1 0.1 0.1];% resolution of one voxel
        size=[100 100 100];      % size of the refractive index
        vector_simulation=true;  % use polarised field or scalar field
        use_abbe_sine=true;
    end
end