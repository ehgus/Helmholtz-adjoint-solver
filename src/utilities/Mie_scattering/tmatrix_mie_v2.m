function [T_ext,T_int,a,b,c,d] = tmatrix_mie_v2(lmax,k_medium,k_particle,radius,mu0,mu1)
%% tmatrix_mie_v2: Make T matrix!

% tmatrix_mie.m : Mie scattering and internal coefficients for uniform
%                 sphere, arranged as a sparse T-matrix.
%
% Usage:
% [T,T2] = tmatrix_mie(Nmax,k_medium,k_particle,radius)
% T is the T-matrix of scattered modes. T2 is the T-matrix of internal
% modes
%
% PACKAGE INFO

n=[1:lmax];

m = k_particle/k_medium;
mmu=mu1/mu0;

r0 = k_medium * radius;
r1 = k_particle * radius;

indexing=combined_index(1:lmax^2+2*lmax)';
%%
% R0 = (ricbesj(n,r0)).';R1 = (ricbesj(n,r1)).';H0 = (ricbesh(n,r0)).';
R0d = ((n+1).*sbesselj(n,r0)-r0.*sbesselj(n+1,r0)).';
R1d = ((n+1).*sbesselj(n,r1)-r1.*sbesselj(n+1,r1)).';
H0d = ((n+1).*sbesselh1(n,r0)-r0.*sbesselh1(n+1,r0)).';
j0 = (sbesselj(n,r0)).';j1 = (sbesselj(n,r1)).';h0 = (sbesselh1(n,r0)).';

%%
% a = -(R1.*R0d - m.*mmu.*R1d.*R0) ./ (R1.*H0d - m.*mmu.*R1d.*H0);
% b = -(R1d.*R0 - m.*mmu.*R1.*R0d) ./ (R1d.*H0 - m.*mmu.*R1.*H0d);
a = -(j0.*R1d - mmu.*j1.*R0d) ./ (h0.*R1d - mmu.*j1.*H0d);
b = -(mmu.*j0.*R1d - m.^2.*j1.*R0d) ./ (mmu.*h0.*R1d - m.^2.*j1.*H0d);
T_ext=sparse([1:2*(lmax^2+2*lmax)],[1:2*(lmax^2+2*lmax)],[a(indexing);b(indexing)]);

if nargout>1
%     c = m.*(R0.*H0d - R0d.*H0) ./ (R1.*H0d - m.*mmu.*R1d.*H0);
%     d = m.*(R0d.*H0 - R0.*H0d) ./ (R1d.*H0 - m.*mmu.*R1.*H0d);
    c = (j0 + a.*h0) ./ j1;
    d = (j0 + b.*h0)./(m.*mmu.*j1);
    T_int=sparse([1:2*(lmax^2+2*lmax)],[1:2*(lmax^2+2*lmax)],[c(indexing);d(indexing)]);
    
end
end