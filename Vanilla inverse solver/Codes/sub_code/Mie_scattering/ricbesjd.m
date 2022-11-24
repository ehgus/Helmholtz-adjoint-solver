function [S] = ricbesjd(nu,z)
%% ricbesjd: Riccati-Bessel function of the first kind - differential form

%RICBESJ Riccati-Bessel function of the first kind.
%   S = RICBESJ(NU,Z) is the Riccati-Bessel function of the first kind for
%   each element of the complex array Z.
%
%   See also:
%   RICBESY, RICBESH, DRICBESJ, DRICBESY, DRICBESH, SBESSELH
%
%   Copyright 2012 Jan Schäfer, Institut für Lasertechnologien (ILM)
%   Author: Jan Schäfer (jan.schaefer@ilm.uni-ulm.de)
%   Organization: Institut für Lasertechnologien in der Medizin und
%       Meßtechnik an der Universität Ulm (http://www.ilm-ulm.de)

S = ((nu+1).*sbesselj(nu,z)-z.*sbesselj(nu+1,z));

end