function [B,C,P] = vec_spherical_harmonics(l,cos_theta,legendre_val, legendre_val_higher)
%% vec_spherical_harmonics: Vector spherical harmonics 

% B, C, P wavevectors developed in phasor
% Bohren & Huhst 1998, Chap 4.2
    B=zeros(2*l+1,length(cos_theta),3,'like',cos_theta);
    C=zeros(2*l+1,length(cos_theta),3,'like',cos_theta);
    P=zeros(2*l+1,length(cos_theta),3,'like',cos_theta);
    if nargin == 4
        [d, d_sintheta, d_dtheta] = sharms(l,cos_theta,legendre_val,legendre_val_higher);
    else
        [d, d_sintheta, d_dtheta] = sharms(l,cos_theta);
    end
    ms=(-l:l)';
% Spherical coordinates
    B(:,:,2)=d_dtheta;
    B(:,:,3)=1i.*ms.*d_sintheta;
    C(:,:,2)=B(:,:,3);
    C(:,:,3)=-d_dtheta;
    P(:,:,1)=d;
    B(isnan(B)) = 0; C(isnan(C)) = 0;P(isnan(P)) = 0;
end