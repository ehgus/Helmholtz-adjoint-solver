function utility = DERIVE_OPTICAL_TOOL(params,use_gpu)
% Generate coordinate-relate parameters
% Any generated arrays will be on GPU if use_gpu is true
if nargin<2
    use_gpu=false;
end
%image space / fourier space coordinate information
utility=struct('image_space',[],'fourier_space',[]);

utility.image_space = struct;
utility.image_space.res = num2cell(params.resolution);
utility.image_space.size = num2cell(params.size);
utility.image_space.coor = cell(1,3);

utility.fourier_space = struct;
utility.fourier_space.res = num2cell(1./(params.resolution.*params.size));
utility.fourier_space.size = num2cell(params.size);
utility.fourier_space.coor = cell(1,3);

space_type_list = {'image_space','fourier_space'};
for space_type_idx = 1:2
    space_type = space_type_list{space_type_idx};
    space_res = utility.(space_type).res;
    space_size = utility.(space_type).size;
    for dim = 1:3
        coor_axis = single(ceil(-space_size{dim}/2):ceil(space_size{dim}/2-1));
        coor_axis = coor_axis*space_res{dim};
        coor_axis = reshape(coor_axis, circshift([1 1 space_size{dim}],dim));
        if use_gpu
            coor_axis = gpuArray(coor_axis);
        end
        utility.(space_type).coor{dim} = coor_axis;
    end
end

utility.fourier_space.coorxy=sqrt(...
    (utility.fourier_space.coor{1}).^2+...
    (utility.fourier_space.coor{2}).^2);
%other
utility.lambda=params.wavelength;
utility.nm=params.RI_bg;
utility.k0=1/params.wavelength;
utility.k0_nm=utility.nm*utility.k0;
utility.kmax=params.NA*utility.k0;
utility.NA_circle=utility.fourier_space.coorxy<utility.kmax;
utility.k3=(utility.k0_nm).^2-(utility.fourier_space.coorxy).^2;
utility.k3(utility.k3<0)=0;
utility.k3=sqrt(utility.k3);
%utility.k3=sqrt(max(0,(utility.k0_nm).^2-(utility.fourier_space.coorxy).^2));
utility.dV=prod([utility.image_space.res{1:3}]);
utility.dVk=1/utility.dV;
utility.refocusing_kernel=2i*pi*utility.k3;
utility.cos_theta=utility.k3/utility.k0_nm;
end

