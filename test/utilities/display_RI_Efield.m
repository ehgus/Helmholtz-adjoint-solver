function display_RI_Efield(forward_solver,RI,input_field,figure_name)
    forward_solver.return_3D = true;
    forward_solver.return_reflection = true;
    forward_solver.return_transmission = true;
    forward_solver.set_RI(RI); % change to RI_optimized and run if you want to see the output of adjoint method
    tic;
    [field_trans,~,field_3D]=forward_solver.solve(input_field(:,:,:,1));
    toc;

    % tranform vector field to scalar field
    [input_field_scalar,field_trans_scalar]=vector2scalarfield(input_field,field_trans);
    input_field_no_zero=input_field_scalar;
    zero_part_mask=abs(input_field_scalar)<=0.01*mean(abs(input_field_scalar),'all');
    input_field_no_zero(zero_part_mask)=0.01*exp(1i.*angle(input_field_no_zero(zero_part_mask)));
    relative_complex_trans_field = field_trans_scalar./input_field_no_zero;
    intensity_map = sum(abs(field_3D).^2,4);
    
    % Display results: transmitted field
    figure('Name',figure_name);colormap parula;
    subplot(2,1,1);imagesc(squeeze(abs(relative_complex_trans_field)));title('Amplitude of transmitted light');
    subplot(2,1,2);imagesc(squeeze(angle(relative_complex_trans_field)));title('Phase of transmitted light');
    figure('Name',[figure_name '- intensity map']);orthosliceViewer(intensity_map);title('amplitude in material');colormap gray
    figure('Name',[figure_name '- real RI map']);orthosliceViewer(real(RI));title('RI of material');colormap gray
    figure('Name',[figure_name '- intensity and RI']);hold on;
    plot(squeeze(real(field_3D(floor(end/2),floor(end/2),:,1))), 'r');
    plot(squeeze(real(RI(floor(end/2),floor(end/2),:))),'b');legend('E field','RI');title('Values along z aixs');
end