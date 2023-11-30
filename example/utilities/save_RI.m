function save_RI(filename, RI, resolution, wavelength, field)
    narginchk(3,5)
    save_args = {'RI','resolution','wavelength','field'};
    save_args = save_args(1:nargin-1);
    % save volume_RI and simulation condition in file_path
    if isfile(filename)
        delete(filename)
    end
    save(filename, save_args{:});
end