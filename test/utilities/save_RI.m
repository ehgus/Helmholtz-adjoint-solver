function save_RI(filename, RI, resolution, wavelength)
    % save volume_RI and simulation condition in file_path
    if isfile(filename)
        delete(filename)
    end
    save(filename, 'RI','resolution','wavelength');
end