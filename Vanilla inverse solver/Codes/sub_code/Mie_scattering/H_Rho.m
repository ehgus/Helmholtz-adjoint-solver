function  h_rho = H_Rho(h,rho,l)
%% H_Rho: sbesselj divided by z.

    h_rho = h./rho;
    if l == 0
        h_rho(isnan(h_rho))=0;
    else
    h_rho(isnan(h_rho)) = sbesselj(l-1, 0)/(l+2);
    end
end