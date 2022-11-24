function dxi_rho = Dxi_Rho(dxi,rho,l)
%% Dxi_Rho: sbesselj differential divided by z

    dxi_rho = dxi./rho;
    if l == 0
        dxi_rho(isnan(h_rho))=0;
    else
    dxi_rho(isnan(dxi_rho)) = 2*sbesselj(l-1, 0)/(l+2);
    end
end