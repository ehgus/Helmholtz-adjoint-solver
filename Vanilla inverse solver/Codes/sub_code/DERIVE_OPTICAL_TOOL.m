function utility = DERIVE_OPTICAL_TOOL(optical_parameter,use_gpu)

params= BASIC_OPTICAL_PARAMETER();
if nargin>0 && ~ isempty(optical_parameter)
    params=update_struct(params,optical_parameter);
end
if nargin<2
    use_gpu=false;
end
utility=struct;

coor_index=cell(3);
coor_index{1}= reshape(cat(2,0:floor((params.size(1)-1)/2),floor(params.size(1)/2):-1:1),[],1,1);
coor_index{2}= reshape(cat(2,0:floor((params.size(2)-1)/2),floor(params.size(2)/2):-1:1),1,[],1);
coor_index{3}= reshape(cat(2,0:floor((params.size(3)-1)/2),floor(params.size(3)/2):-1:1),1,1,[]);
%image space
utility.image_space=struct;
utility.image_space.res=params.resolution(1:3);
utility.image_space.size=params.size(1:3);
utility.image_space.coor=coor_index;
utility.image_space.coor{1}=utility.image_space.coor{1}.*utility.image_space.res(1);
utility.image_space.coor{2}=utility.image_space.coor{2}.*utility.image_space.res(2);
utility.image_space.coor{3}=utility.image_space.coor{3}.*utility.image_space.res(3);
%fourier space
utility.fourier_space=struct;
utility.fourier_space.res=1./(utility.image_space.res.*utility.image_space.size);
utility.fourier_space.size=utility.image_space.size;
utility.fourier_space.coor=coor_index;
utility.fourier_space.coor{1}=utility.fourier_space.coor{1}.*utility.fourier_space.res(1);
utility.fourier_space.coor{2}=utility.fourier_space.coor{2}.*utility.fourier_space.res(2);
utility.fourier_space.coor{3}=utility.fourier_space.coor{3}.*utility.fourier_space.res(3);
utility.fourier_space.coorxy=sqrt(...
    (utility.fourier_space.coor{1}).^2+...
    (utility.fourier_space.coor{2}).^2);
%other
utility.lambda=params.wavelength;
utility.k0=1/params.wavelength;
utility.k0_nm=utility.k0.*params.RI_bg;
utility.nm=params.RI_bg;
utility.kmax=params.NA/params.wavelength;
utility.NA_circle=utility.fourier_space.coorxy<utility.kmax;
utility.k3=(utility.k0_nm).^2-(utility.fourier_space.coorxy).^2;utility.k3(utility.k3<0)=0;utility.k3=sqrt(utility.k3);
utility.dV=prod(utility.image_space.res);
utility.dVk=1/utility.dV;
utility.refocusing_kernel=2i*pi*utility.k3;
utility.cos_theta=utility.k3/utility.k0_nm;
%move to the gpu the needed arrays (scalar are kept on cpu)
if use_gpu
    utility.image_space.coor{1}=gpuArray(utility.image_space.coor{1});
    utility.image_space.coor{2}=gpuArray(utility.image_space.coor{2});
    utility.image_space.coor{3}=gpuArray(utility.image_space.coor{3});
    utility.fourier_space.coor{1}=gpuArray(utility.fourier_space.coor{1});
    utility.fourier_space.coor{2}=gpuArray(utility.fourier_space.coor{2});
    utility.fourier_space.coor{3}=gpuArray(utility.fourier_space.coor{3});
    utility.fourier_space.coorxy=gpuArray(utility.fourier_space.coorxy);
    utility.NA_circle=gpuArray(utility.NA_circle);
    utility.k3=gpuArray(utility.k3);
    utility.refocusing_kernel=gpuArray(utility.refocusing_kernel);
    utility.cos_theta=gpuArray(utility.cos_theta);
end

end

