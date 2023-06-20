function bow_repr = computeBOWRepr( features, means)
[k,~] = size(means);
bow = zeros( 1, k );
[num_features, ~] = size( features );

for i = 1:num_features
    feature = features(i,:);
    if max(feature) > 0
        index = compute_distance_cluster( feature, means);
        bow(index) = bow(index) + 1;
    end
end

bow_repr = bow/sum(bow);

function index = compute_distance_cluster( feature, means)
distance = means*feature';
[~,index] = min(distance);
