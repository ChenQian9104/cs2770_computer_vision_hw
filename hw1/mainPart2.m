close all
image = imread('cardinal1.jpg'); 
filters = load('filters.mat'); 
filters = filters.F;
F = filters;

[ texture_repr_concat, texture_repr_mean] = computeTextureReprs( image, F );
    
