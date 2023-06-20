function index = compute_distance_cluster( feature, means)
distance = means*feature';
[~,index] = min(distance);
