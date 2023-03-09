function [Field, FoM] =solve_adjoint(h,conjugate_field, options)
    % Generate adjoint source and calculate adjoint field
    if (h.forward_solver.use_GPU)
        conjugate_field = gpuArray(conjugate_field);
    end
    % pre-allocate empty array
    Nsize = h.forward_solver.size + 2 * h.forward_solver.boundary_thickness_pixel;
    if h.forward_solver.vector_simulation
        Nsize(4) = 3;
    end
    if h.forward_solver.use_GPU
        adjoint_field = zeros(Nsize,'single','gpuArray');
    else
        adjoint_field = zeros(Nsize,'single');
    end
    if h.mode == "Intensity"
        FoM = - sum(abs(conjugate_field).^2.*options.intensity_weight,'all') / numel(conjugate_field);
        adjoint_field(h.forward_solver.ROI(1):h.forward_solver.ROI(2),h.forward_solver.ROI(3):h.forward_solver.ROI(4),h.forward_solver.ROI(5):h.forward_solver.ROI(6),:) = - conjugate_field.*options.intensity_weight;
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
        field_interest = conjugate_field(ROI_field(1):ROI_field(2), ROI_field(3):ROI_field(4), ROI_field(5):ROI_field(6), :);
        field_interest = ifft2(field_interest);
        Nsize = size(conjugate_field);
        z_padding = h.forward_solver.boundary_thickness_pixel(3);
        Nsize(3) = Nsize(3) + 2*z_padding;
        % find ROI of field
        x_idx = options.x_idx;
        y_idx = options.y_idx;
        field_of_range = field_interest(x_idx, y_idx, :, :);
        % calculate transmission profile
        % kx^2 + ky^2 + kz^2 = k^2 = (2pi*n/wavelength)^2
        % axial_propagation  = exp(1i*kz*z) = exp(1i*kz*dz*n)
        axial_propagation = options.axial_propagation;
        field_of_range = field_of_range .* axial_propagation(:,:,z_padding+ROI_field(5):z_padding+ROI_field(6));
        %subpixel correction is requried: future
        sub_pixel_phase = exp(-1i .* reshape(1:Nsize(3),1,1,[]) .* angle(mean(field_of_range(:,:,2:end,:)./field_of_range(:,:,1:end-1,:),3)));
        field_of_range = field_of_range.*sub_pixel_phase(:,:,z_padding+ROI_field(5):z_padding+ROI_field(6),:);
        % get value
        transmission_profile = sum(field_of_range, 3)/size(field_of_range, 3);
        phase = exp(1i * angle(transmission_profile));
        % generate adjoint source
        transmission_profile = options.relative_intensity.*phase - transmission_profile;
        transmission_profile(isnan(transmission_profile)) = 0;
        adjoint_field(x_idx, y_idx, :, :) = transmission_profile.*conj(axial_propagation.*sub_pixel_phase);
        adjoint_field = fft2(adjoint_field);
        % figure of merit
        FoM = sum(abs(transmission_profile).^2,'all');
    end
    % Evaluate output field
    Field = h.forward_solver.eval_scattered_field(adjoint_field);
    Field = Field + gather(adjoint_field(h.forward_solver.ROI(1):h.forward_solver.ROI(2),h.forward_solver.ROI(3):h.forward_solver.ROI(4),h.forward_solver.ROI(5):h.forward_solver.ROI(6),:));
end
