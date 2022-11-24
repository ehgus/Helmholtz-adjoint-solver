function    [d, d_sintheta, d_dtheta] = sharms(l,x)
%% sharms: Legendre functions used for vsh

    if size(x,1)>size(x,2)
        x=x';
    end
    legendre_total=(associated_Legendre(l,x)); % [R, C] = [m, x] / |m| <= l, l order
    
    ms=(-l:1:l)';
    
% 1. Obtain d
    norm_legendre=sqrt(factorial(l-ms)./factorial(l+ms));
    d = legendre_total.*norm_legendre;
% 2. Obtain d_sintheta
    legendre_higher = associated_Legendre(l+1,x); legendre_lower_1 = legendre_higher(3:end,:);
    legendre_lower_2= legendre_higher(1:end-2,:);
    plm_sintheta = -1./(2.*ms) .* (legendre_lower_1 + (l-ms+1).*(l-ms+2).*legendre_lower_2);
    plm_sintheta(ms == 0,:) = legendre_total(ms == 0, :) ./ sqrt(1-x.^2);
    plm_sintheta(ms == 0,abs(x)==1) = 0;
    d_sintheta = plm_sintheta .* norm_legendre;
% 3. Obtain d_dtheta
    legendre_lower_1 = legendre_total - legendre_total;
    legendre_lower_1(1:end-1,:) = legendre_total(2:end,:);
    plm_dtheta = legendre_lower_1 + ms.*x.*plm_sintheta;
    d_dtheta = plm_dtheta .* norm_legendre; 
end