function [S] = ricbesh1d(nu,z)
%RICBESH Riccati-Hankel function of the first kind.
%   S = RICBESH(NU,Z) is the Riccati-Hankel function of the first kind for
%   each element of the complex array Z.
%
%   See also:
%   RICBESY, RICBESH, DRICBESJ, DRICBESY, DRICBESH, SBESSELH
%
%   Copyright 2012 Jan Schäfer, Institut für Lasertechnologien (ILM)
%   Author: Jan Schäfer (jan.schaefer@ilm.uni-ulm.de)
%   Organization: Institut für Lasertechnologien in der Medizin und
%       Meßtechnik an der Universität Ulm (http://www.ilm-ulm.de)

S = ((nu+1).*sbesselh1(nu,z)-z.*sbesselh1(nu+1,z));

end