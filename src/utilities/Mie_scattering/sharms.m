function    [d, d_sintheta, d_dtheta] = sharms(l,x,legendre_val,legendre_val_higher)
%% sharms: Legendre functions used for vsh

    if size(x,1)>size(x,2)
        x=x';
    end
    if nargin < 4
        legendre_val=associated_Legendre(l,x);
        legendre_val_higher = associated_Legendre(l+1,x);
    end
    ms=(-l:l)';
    
% 1. Obtain d
    norm_legendre=sqrt(factorial(l-ms)./factorial(l+ms));
    d = legendre_val.*norm_legendre;
% 2. Obtain d_sintheta
    plm_sintheta = -1./(2.*ms) .* (legendre_val_higher(3:end,:) + (l-ms+1).*(l-ms+2).*legendre_val_higher(1:end-2,:));
    plm_sintheta(ms == 0,:) = legendre_val(ms == 0, :) ./ sqrt(1-x.^2);
    plm_sintheta(ms == 0,abs(x)==1) = 0;
    d_sintheta = plm_sintheta .* norm_legendre;
% 3. Obtain d_dtheta
    plm_dtheta = zeros(size(legendre_val),'like',legendre_val);
    plm_dtheta(1:end-1,:) = legendre_val(2:end,:);
    plm_dtheta = plm_dtheta + ms.*x.*plm_sintheta;
    d_dtheta = plm_dtheta .* norm_legendre; 
end