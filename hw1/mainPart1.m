clc;clear;
data = load('filters.mat');
filters = data.F;


cardinal_1 = imread('cardinal1.jpg');
cardinal_1 = rgb2gray( cardinal_1 );
cardinal_1 = imresize( cardinal_1, [100, 100] );


cardinal_2 = imread('cardinal2.jpg');
cardinal_2 = rgb2gray( cardinal_2 );
cardinal_2 = imresize( cardinal_2, [100, 100] );

leopard_1 = imread('leopard1.jpg');
leopard_1 = rgb2gray( leopard_1 );
leopard_1 = imresize( leopard_1, [100, 100] );

leopard_2 = imread('leopard2.jpg');
leopard_2 = rgb2gray( leopard_2 );
leopard_2 = imresize( leopard_2, [100, 100] );


panda_1 = imread('panda1.jpg');
panda_1 = rgb2gray( panda_1 );
panda_1 = imresize( panda_1, [100, 100] );

panda_2 = imread('panda2.jpg');
panda_2 = rgb2gray( panda_2 );
panda_2 = imresize( panda_2, [100, 100] );


%% =============== Same animals similar ============================%%
i = 40; figure(1)

filter = filters(:,:,i);
subplot( 2, 4, 1)
imagesc(filter); 


subplot( 2, 4, 3)
filter_image = imfilter( cardinal_1, filter, 'conv');
imagesc(filter_image); 
title('cardinal1.jpg');

subplot( 2, 4, 4)
filter_image = imfilter( cardinal_2, filter, 'conv');
imagesc(filter_image); 
title('cardinal2.jpg');

subplot( 2, 4, 5)
filter_image = imfilter( leopard_1, filter, 'conv');
imagesc(filter_image); 
title('leopard1.jpg');

subplot( 2, 4, 6)
filter_image = imfilter( leopard_2, filter, 'conv');
imagesc(filter_image); 
title('leopard2.jpg');

subplot( 2, 4, 7)
filter_image = imfilter( panda_1, filter, 'conv');
imagesc(filter_image); 
title('panda1.jpg');

subplot( 2, 4, 8)
filter_image = imfilter( panda_2, filter, 'conv');
imagesc(filter_image); 
title('panda2.jpg');






%% =============== Different animals similar ============================%%
i = 10; figure(2)

filter = filters(:,:,i);
subplot( 2, 4, 1)
imagesc(filter); 


subplot( 2, 4, 3)
filter_image = imfilter( cardinal_1, filter, 'conv');
imagesc(filter_image); 
title('cardinal1.jpg');

subplot( 2, 4, 4)
filter_image = imfilter( cardinal_2, filter, 'conv');
imagesc(filter_image); 
title('cardinal2.jpg');

subplot( 2, 4, 5)
filter_image = imfilter( leopard_1, filter, 'conv');
imagesc(filter_image); 
title('leopard1.jpg');

subplot( 2, 4, 6)
filter_image = imfilter( leopard_2, filter, 'conv');
imagesc(filter_image); 
title('leopard2.jpg');

subplot( 2, 4, 7)
filter_image = imfilter( panda_1, filter, 'conv');
imagesc(filter_image); 
title('panda1.jpg');

subplot( 2, 4, 8)
filter_image = imfilter( panda_2, filter, 'conv');
imagesc(filter_image); 
title('panda2.jpg');



