function display_RI_Efield(forward_solver,RI,current_source,figure_name)
    forward_solver.set_RI(RI); % change to RI_optimized and run if you want to see the output of adjoint method
    tic;
    Efield = forward_solver.solve(current_source(:,:,:,1));
    toc;

    % tranform vector field to scalar field
    intensity_map = sum(abs(Efield).^2,4);
    
    % Display results
    figure('Name',[figure_name '- intensity map']);orthosliceViewer(intensity_map);title('amplitude in material');colormap gray
    figure('Name',[figure_name '- real RI map']);orthosliceViewer(real(RI));title('RI of material');colormap gray
    figure('Name',[figure_name '- E field and RI']);hold on;
    plot(squeeze(real(Efield(floor(end/2),floor(end/2),:,1))), 'r');
    plot(squeeze(real(RI(floor(end/2),floor(end/2),:))),'b');legend('E field','RI');title('Values along z aixs');
end