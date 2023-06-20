function M = intensity_change( Ix,Iy, r, c, offset) 
M = zeros(2,2);

for i = -offset: 1 : offset
    for j = -offset:1:offset
        M(1,1) = M(1,1) + Ix( r + i, c + j )^2;
        M(1,2) = M(1,2) + Ix( r + i, c + j )*Iy( r + i, c + j);
        M(2,1) = M(2,1) + Ix( r + i, c + j )*Iy( r + i, c + j);
        M(2,2) = M(2,2) + Iy( r + i, c + j )^2;
    end
end