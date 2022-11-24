function legendre_total = associated_Legendre(l,x)
%% associated_Legendre

    legendre_total=zeros(2*l+1,length(x));
    legendre_total(l+1:end,:)=legendre(l,x);
    ms=(1:l)';
    legendre_total(1:l,:)=flipud(legendre_total(l+2:end,:).*repmat((-1).^ms.*factorial(l-ms)./factorial(l+ms),[1 length(x)]));
end