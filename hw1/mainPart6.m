means = load('means.mat');
means = means.means;
bow_repr = computeBOWRepr( features, means);
