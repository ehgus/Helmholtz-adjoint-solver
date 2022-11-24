function [R] = getTransformationMatrix( kphi, ktheta )
%% getTransformationMatrix: Transformation matrix from (theta, phi) to unit (x,y,z)!

%GETTRANSFORMATIONMATRIX Returns the matrix to transform a 3D vector from
%   spherical/cylindrical coordinate basis to cartesian coordinate basis.
%
%   [R] = NFCYL(PHI,THETA) returns the 3x3 transformation matrix to express
%   a 3D vector in cartesian coordinate basis. If PHI and THETA is given
%   the original vector basis is assumed to be in spherical coordinates, if
%   only PHI is submitted original vector basis is in cylindrical
%   coordinates.
%
%   Copyright 2012 Jan Schäfer, Institut für Lasertechnologien (ILM)
%   Author: Jan Schäfer (jan.schaefer@ilm.uni-ulm.de)
%   Organization: Institut für Lasertechnologien in der Medizin und
%       Meßtechnik an der Universität Ulm (http://www.ilm-ulm.de)

R = zeros(3,3,length(kphi));

if ~exist('ktheta', 'var')
    R(1,1,:) = cos(kphi);
    R(1,2,:) = -sin(kphi);
    R(1,3,:) = 0.;
    R(2,1,:) = sin(kphi);
    R(2,2,:) = cos(kphi);
    R(2,3,:) = 0.;
    R(3,1,:) = 0.;
    R(3,2,:) = 0.;
    R(3,3,:) = 1.;
else %~exist('theta', 'var')
    R(1,1,:) = sin(ktheta).*cos(kphi);
    R(1,2,:) = cos(ktheta).*cos(kphi);
    R(1,3,:) = -sin(kphi);
    R(2,1,:) = sin(ktheta).*sin(kphi);
    R(2,2,:) = cos(ktheta).*sin(kphi);
    R(2,3,:) = cos(kphi);
    R(3,1,:) = cos(ktheta);
    R(3,2,:) = -sin(ktheta);
    R(3,3,:) = 0;
end %if ~exist('theta', 'var')

end