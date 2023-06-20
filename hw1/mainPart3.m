close all
im1 = imread('baby_happy.jpg');
im2 = imread('baby_weird.jpg');

im1 = im2double(im1);
im2 = im2double(im2); 

im1 = rgb2gray(im1);
im1 = imresize( im1, [512, 512]);

im2 = rgb2gray(im2);
im2 = imresize( im2, [512, 512]);

im1_blur = imgaussfilt(im1, 5);
im2_blur = imgaussfilt(im2, 5);

im2_detail = im2 - im2_blur;


im_hybrid = im1_blur + im2_detail;
figure(1); imagesc(im_hybrid)