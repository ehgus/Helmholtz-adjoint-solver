function [spol, ppol] = parallel_transport_pol(kvecs)
%% parallel_transport_pol

% Equation reference: Eq. (11), which is derived from parallel transport law
% Link  :  https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-18-4-839
    [ktheta,kphi,~] = xcart2sph(kvecs(:,1),kvecs(:,2),kvecs(:,3));
        kvecs2 = kvecs./sqrt(kvecs(:,1).^2+kvecs(:,2).^2+kvecs(:,3).^2);
   
    pol_new = zeros(size(kvecs));
    pol_new(:,1) = cos(ktheta).*cos(kphi).^2+sin(kphi).^2;
    pol_new(:,2) = -sin(kphi).*cos(kphi).*(1-cos(ktheta));
    pol_new(:,3) = -cos(kphi).*sin(ktheta);

    spol = pol_new;
    ppol = (cross(kvecs2, spol));
end