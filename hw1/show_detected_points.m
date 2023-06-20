function show_detected_points(image,x,y, scores)
figure()
imshow(image); hold on; 

for i = 1:max( size(x) )
    scatter(x(i), y(i), 0.5*scores(i), 'filled' );
end

