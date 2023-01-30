function [RI, resolution, wavelength] = load_RI(filename)
    % save volume_RI and simulation condition in file_path (HDF5 format)
    if ~isfile(filename)
        error('file does not exist')
    end
    load(filename, 'RI','resolution','wavelength');
end