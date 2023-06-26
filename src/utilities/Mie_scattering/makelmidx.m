function [n,m]=makelmidx(lmax)
%% makelmidx: Generate indices for Mie scattering coefficients, (n,m)
n=zeros(lmax^2-1,1);m=zeros(lmax^2-1,1);
for l_val=1:lmax
    n(l_val^2:(l_val+1)^2-1)=l_val;
    m(l_val^2:(l_val+1)^2-1)=(-l_val:1:l_val)';
end

end