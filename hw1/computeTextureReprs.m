function [ texture_repr_concat, texture_repr_mean] = ...
    computeTextureReprs( image, F )


image = rgb2gray(image);
[~,~,num_filters] = size(F);
[num_rows, num_cols] = size(image);
response = zeros( num_filters, num_rows, num_cols );     % create a new variable: response

texture_repr_mean   = zeros( num_filters,1);
for i = 1: num_filters
    response(i, :, : ) = imfilter( image, F(:,:,i) );
    texture_repr_mean(i) = sum( sum( response(i,:,:) ) )/( num_rows*num_cols );
end

texture_repr_concat = response(:);

