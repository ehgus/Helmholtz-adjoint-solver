function RI_opt=solve(h,input_field,RI,options)
% Adjoint solver targeting PDMS+TiO2+Microchem SU-8 2000 layered material.
% It is not a universal solver.

% set parameters
RI_opt=single(RI);
Figure_of_Merit=zeros(h.itter_max,1);

h.optimizer.reset();
% preprocess update parameter
options = h.preprocess_params(options);

% Run!
h.gradient = complex(zeros(size(RI,1:4),'single'));
h.density_map = zeros(size(RI,1:3),'single');

gradient_full_size = size(RI,1:5);
if size(input_field,3) ==2
    gradient_full_size(4) = 3;
end
gradient_full_size(5) = size(input_field,4);
h.gradient_full = complex(zeros(gradient_full_size,'single'));

for ii=1:h.itter_max
    display(['Iteration: ' num2str(ii)]);
    tic;
    
    % Calculated gradient RI based on intensity mode
    h.forward_solver.set_RI(RI_opt);
    [E_fwd, H_fwd] = h.forward_solver.solve(input_field);
    [E_adj, Figure_of_Merit(ii)]=h.solve_adjoint(E_fwd, H_fwd, options);
    h.get_gradient(E_adj, E_fwd, RI_opt, ii);

    RI_opt = h.optimizer.apply_gradient(RI_opt, RI_opt, h.gradient, h.step);
    RI_opt=h.post_regularization(RI_opt,ii);
    h.RI_inter=RI_opt;
    toc;
    if h.verbose
        figure(201)
        plot(Figure_of_Merit(1:ii));
        drawnow;
    end
end

RI_opt=gather(RI_opt);
end