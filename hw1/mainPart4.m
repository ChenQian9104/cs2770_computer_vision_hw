close all;
image = 'panda1.jpg';
image = imread(image);
[ x,y, scores, Ix, Iy ] = extract_keypoints(image);
image_gray = rgb2gray(image);
show_detected_points(image_gray,x,y, scores);


image = 'leopard1.jpg';
image = imread(image);
[ x,y, scores, Ix, Iy ] = extract_keypoints(image);
image_gray = rgb2gray(image);
show_detected_points(image_gray,x,y, scores);


image = 'cardinal1.jpg';
image = imread(image);
[ x,y, scores, Ix, Iy ] = extract_keypoints(image);
image_gray = rgb2gray(image);
show_detected_points(image_gray,x,y, scores);