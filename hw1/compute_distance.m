function d = compute_distance( X, Y )
d = 0;
for i = 1: max( size(X) )
    d = d + ( X(i) - Y(i) )^2;
end
d = sqrt(d);