function [Field, FoM] =solve_adjoint(h,conjugated_field, options)
    % Generate adjoint source and calculate adjoint field
    if (h.forward_solver.use_GPU)
        conjugated_field=gpuArray(single(conjugated_field));
    end

    if h.mode == "Intensity"
        adjoint_field = conjugated_field.*options.intensity_weight;
        FoM = sum(abs(adjoint_field).^2.*options.intensity_weight,'all') / numel(adjoint_field);
    elseif h.mode == "Transmission"
        % Calculate transmission rate of plane wave
        % relative intensity: Matrix of relative intensity
        %   - NaN: do not consider in calculation
        %   - value: desired phase and intensity
        % diffraction_order: the range of diffraction order of interest
        %   - diffraction_order.x: the range of diffraction order of interest in x axis
        %   - diffraction_order.y: the range of diffraction order of interest in y axis
        % ROI_field: area that will decompose field
        ROI_field = options.ROI_field;
        field_of_range = conjugated_field(ROI_field(1):ROI_field(2), ROI_field(3):ROI_field(4), ROI_field(5):ROI_field(6), :);
        field_of_range = ifft2(field_of_range);
        % find ROI of field
        diffraction_order = options.diffraction_order;
        x_idx = diffraction_order.x(1) + 1:diffraction_order.x(2) + 1;
        x_idx(x_idx<=0) = x_idx(x_idx<=0) + size(field_of_range, 1);
        y_idx = diffraction_order.y(1) + 1:diffraction_order.y(2) + 1;
        y_idx(y_idx<=0) = y_idx(y_idx<=0) + size(field_of_range, 2);
        field_of_range = field_of_range(x_idx, y_idx, :, :);
        % calculate transmission profile
        % kx^2 + ky^2 + kz^2 = k^2 = (2pi*n/wavelength)^2
        % axial_propagation  = exp(1i*kz*z) = exp(1i*kz*dz*n)
        Nsize = size(conjugated_field);
        %x_freq = reshape(diffraction_order.x(1):diffraction_order.x(2),[],1)/(h.resolution(1)*Nsize(1));
        %y_freq = reshape(diffraction_order.y(1):diffraction_order.y(1),1,[])/(h.resolution(2)*Nsize(2));
        %kz = 2*pi*sqrt((h.forward_solver.RI_bg/h.wavelength)^2 - (x_freq.^2 + y_freq.^2));
        %axial_propagation = conj(exp(-1i*h.resolution(3).*kz.*reshape(1:Nsize(3),1,1,Nsize(3))));
        axial_propagation = h.forward_solver.refocusing_util(x_idx, y_idx, :);
        field_of_range = field_of_range .* axial_propagation(:,:,ROI_field(5):ROI_field(6));
        transmission_profile = sum(field_of_range, 3)/size(field_of_range, 3);
        magnitude = abs(transmission_profile);
        phase = transmission_profile./magnitude;
        relative_intensity = reshape(options.relative_intensity, numel(x_idx), numel(y_idx), 1, 3);
        % generate adjoint source
        transmission_profile = (magnitude - relative_intensity).*phase; % order does not matched
        disp(abs(transmission_profile(:,:,:,1)))
        transmission_profile(isnan(transmission_profile)) = 0;
        adjoint_field = zeros(Nsize,'like',conjugated_field);
        adjoint_field(x_idx, y_idx, :, :) = transmission_profile.*conj(axial_propagation);
        adjoint_field = fft2(adjoint_field);
        % figure of merit
        FoM = sum(abs(transmission_profile).^2,'all');
    end
    % Evaluate output field
    Field = h.forward_solver.eval_scattered_field(adjoint_field);
    Field = Field + adjoint_field;
end