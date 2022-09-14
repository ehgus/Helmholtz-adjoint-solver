function RI_opt=solve(h,input_field,Target_intensity,RI)
% Adjoint solver targeting PDMS+TiO2+Microchem SU-8 2000 layered material.
% It is not a universal solver.

assert(mod(size(RI,3),2)==1, 'Length of RI block along z axis should be odd');
assert(strcmp(h.parameters.mode, "Intensity"),"Transmission mode is not implemented yet")

% set parameters
RI_opt=single(RI);
Figure_of_Merit=zeros(h.parameters.itter_max,1);
alpha=1/h.parameters.step;

s_n=0;
t_n=0;
t_np=1;

u_n=RI_opt.^2;
x_n=RI_opt.^2; 

nmin = h.nmin; % RI of PDMS 
nmax = h.nmax; % RI of TiO2

% Run!

h.gradient = zeros(size(RI,1:4),'single');

gradient_full_size = size(RI,1:5);
if size(input_field,3) ==2
    gradient_full_size(4) = 3;
end
gradient_full_size(5) = size(input_field,4);
h.gradient_full = zeros(gradient_full_size,'single');
isRItensor = size(RI,4) == 3;

for ii=1:h.parameters.itter_max
    display(['Iteration: ' num2str(ii)]);
    tic;
    
    % Calculated gradient RI based on intensity mode
    h.forward_solver.set_RI(RI_opt);
    [~,~,E_old]=h.forward_solver.solve(input_field);
    h.forward_solver.set_RI((flip(RI_opt,3))); % flip verison
    E_adj=h.solve_adjoint(flip(conj(E_old),3),flip(sqrt(Target_intensity),3));
    E_adj=flip(E_adj,3);
    Figure_of_Merit(ii) = sum(abs(E_old).^2.*Target_intensity,'all') / sum(abs(E_old).^2,'all');
    
    h.get_gradeint(E_adj,E_old,isRItensor);
    
    % update the result
    % The update method is based on FISTA algorithm
    t_n=t_np;
    t_np=(1+sqrt(1+4*t_n^2))/2;
    % maximization (negative sign)
    s_n = u_n-(1/alpha)*h.gradient;
    s_n=gather(s_n);

    u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
    x_n=s_n;

    RI_opt=sqrt(u_n);
    RI_opt((h.parameters.ROI_change(:)).*(real(RI_opt(:)) > real(nmax))==1) = nmax;
    RI_opt((h.parameters.ROI_change(:)).*(real(RI_opt(:)) < real(nmin))==1) = nmin;
    h.RI_inter=RI_opt;

    toc;
end

RI_opt=gather(RI_opt);
end