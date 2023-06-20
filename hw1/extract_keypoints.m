function [ x,y, scores, Ix, Iy ] = extract_keypoints(image)


%% ==========================  step 1  ==================================%%
k = 0.05;  % empirical constant
window_size = 5; 
image2detect = im2double(image);
image2detect = rgb2gray( image2detect );

[num_rows, num_cols ] = size( image2detect);

Ix = imfilter( image2detect, [ -1 0 1; -1 0 1; -1 0 1]  );
Iy = imfilter( image2detect, [ -1 0 1; -1 0 1; -1 0 1]' );
R = zeros( num_rows, num_cols);

%% ==========================  step 2 ==================================%%
offset = floor( window_size/2 );

for i = ( 1 + offset):  ( num_rows - offset )
    for j = ( 1 + offset) : ( num_cols - offset)
        M = intensity_change( Ix, Iy, i, j , offset);
        R(i,j) = det(M) - k*trace(M)^2;
    end
end

threshold = 5*sum( sum(R) )/(num_cols*num_rows);
x = []; y = []; scores = [];

for i = ( 1 + offset):  ( num_rows - offset )
    for j = ( 1 + offset) : ( num_cols - offset)
        if R(i,j) >= threshold
            neighbors = [ R(i-1,j-1), R(i-1,j), R(i-1, j + 1),...
                R(i,j-1), R(i,j+1), R(i+1, j-1), R(i+1,j), R(i+1,j+1)];
            if R(i,j) > max( neighbors)
                x = [x j];
                y = [y i];
                scores = [scores R(i,j) ];
            end 
        end
    end
end


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


