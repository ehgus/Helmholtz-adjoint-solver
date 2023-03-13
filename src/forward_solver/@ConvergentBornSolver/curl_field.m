function rst = curl_field(obj, field)
    % curl operator
    assert(size(field,4)==3,"The field should be vectorial");
    [X,Y,Z] = meshgrid(obj.utility.image_space.coor{:});
    rst = curl(X,Y,Z,field(:,:,:,1),field(:,:,:,2),field(:,:,:,3));
end