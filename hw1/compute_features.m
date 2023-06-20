function features = compute_features( x, y, scores, Ix, Iy)
d = 8;
n = max( size(x) ); 
features = zeros( n, d ); 
[num_rows, num_cols ] = size( Ix);


for i = 1:n
    r = y(i); c = x(i);
    
    if ( r > 5 && r < num_rows - 5 ) && ( c > 5 && c < num_cols - 5 )
        [M,theta] = compute_gradient( Ix, Iy, r, c);
        X = collect_bins( M, theta );
        features(i,:) = Normalize_clip( X, 0.2 );
        
    end
    
end



function [M, theta] = compute_gradient(Ix, Iy, r, c)
M = [];theta = [];

for i = -5:1:5
    for j = -5:1:5
        m = sqrt( Ix( r + j, c + i )^2 + Iy( r + j, c + i )^2 );
        if m > 0
            M = [M,m];
            theta = [theta, radtodeg( atan( Iy( r + j, c + i )/Ix( r + j, c + i )) ) ];
        end
    end
end


function X = collect_bins( M, theta )
X = zeros(1,8); 
for i = 1: max( size(theta) )
    if theta(i) <= -67.5
        X(1) = X(1) + M(i);
    elseif theta(i) <= -45
        X(2) = X(2) + M(i);
    elseif theta(i) <= -22.5
        X(3) = X(3) + M(i);
    elseif theta(i) <= 0
        X(4) = X(4) + M(i);
    elseif theta(i) <= 22.5
        X(5) = X(5) + M(i);
    elseif theta(i) <= 45
        X(6) = X(6) + M(i);
    elseif theta(i) <= 67.5       
        X(7) = X(7) + M(i);   
    else
        X(8) = X(8) + M(i);       
    end
    
    
end


function X = Normalize_clip( X, threshold )
X = X/sum(X); 

for i = 1: max( size(X) )
    if X(i) > threshold
        X(i) = threshold; 
    end
end

X = X/sum(X); 
