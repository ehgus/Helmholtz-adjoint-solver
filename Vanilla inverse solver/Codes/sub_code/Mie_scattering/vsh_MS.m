function [B,C,P] = vsh_MS(l,ktheta)
%% vsh_MS: Vector spherical harmonics 

% B, C, P wavevectors developed in phasor
% Bohren & Huhst 1998, Chap 4.2
    B=zeros(2*l+1,length(ktheta),3);C=B;P=B;
    [d, d_sintheta, d_dtheta] = sharms(l,cos(ktheta));
    ms=(-l:1:l)';
% Spherical coordinates
    B(:,:,2)=d_dtheta;                   B(:,:,3)=1i.*ms.*d_sintheta;
    C(:,:,2)=1i.*ms.*d_sintheta;     C(:,:,3)=-d_dtheta;
    P(:,:,1)=d;
    
end