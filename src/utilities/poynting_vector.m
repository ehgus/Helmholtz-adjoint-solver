function S = poynting_vector(E, H)
    % Calculate complex poynting vector: E x conj(H)
    S = cross(E, conj(H), 4);
end