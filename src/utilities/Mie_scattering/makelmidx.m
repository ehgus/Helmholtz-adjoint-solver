function [n,m]=makelmidx(lmax)
%% makelmidx: Generate indices for Mie scattering coefficients, (n,m)
n=zeros(lmax^2-1,1);m=zeros(lmax^2-1,1);
for i1=1:lmax
    n(i1^2:(i1+1)^2-1)=i1;
    m(i1^2:(i1+1)^2-1)=(-i1:1:i1)';
end

end