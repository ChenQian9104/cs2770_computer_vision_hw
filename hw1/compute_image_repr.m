function [bow_repr, texture_repr_concat, texture_repr_mean] = compute_image_repr( image, means, filters)
[ x,y, scores, Ix, Iy ] = extract_keypoints( image );
features = compute_features( x, y, scores, Ix, Iy);
bow_repr = computeBOWRepr( features, means);

bow_repr = bow_repr/sqrt( bow_repr*bow_repr');

[ texture_repr_concat, texture_repr_mean] = computeTextureReprs( image, filters );

texture_repr_concat = texture_repr_concat/sqrt( texture_repr_concat'*texture_repr_concat );
texture_repr_mean = texture_repr_mean/sqrt( texture_repr_mean'*texture_repr_mean );


% texture_repr_concat = texture_repr_concat';
% texture_repr_mean = texture_repr_mean';