function RI_opt=solve(obj, input_field, RI, options)
% Adjoint solver targeting PDMS+TiO2+Microchem SU-8 2000 layered material.
% It is not a universal solver.

% set parameters
RI_opt=single(RI);
Figure_of_Merit=zeros(obj.itter_max,1);

obj.optimizer.reset();
% preprocess update parameter
options = obj.preprocess_params(options);

% Run!
obj.gradient = complex(zeros(size(RI,1:4),'single'));
obj.density_map = zeros(size(RI,1:3),'single');

gradient_full_size = size(RI,1:5);
if size(input_field,3) ==2
    gradient_full_size(4) = 3;
end
gradient_full_size(5) = size(input_field,4);
obj.gradient_full = complex(zeros(gradient_full_size,'single'));

for idx = 1:obj.itter_max
    display(['Iteration: ' num2str(idx)]);
    tic;
    
    % Calculated gradient RI based on intensity mode
    obj.forward_solver.set_RI(RI_opt);
    [E_fwd, H_fwd] = obj.forward_solver.solve(input_field);
    [E_adj, Figure_of_Merit(idx)]=obj.solve_adjoint(E_fwd, H_fwd, options);
    obj.get_gradient(E_adj, E_fwd, RI_opt, idx);

    RI_opt = obj.optimizer.apply_gradient(RI_opt, RI_opt, obj.gradient, obj.step);
    RI_opt=obj.post_regularization(RI_opt,idx);
    obj.RI_inter=RI_opt;
    toc;
    if obj.verbose
        figure(201)
        plot(Figure_of_Merit(1:idx));
        drawnow;
    end
end

RI_opt=gather(RI_opt);
end