RGB_indexes = [3,2,1]
inputImage1 = load('.\11.mat');
I_AIHS = inputImage1.result
figure
th_MSrgb = image_quantile(I_AIHS(:,:,RGB_indexes), [0.01 0.99]);
imshow(image_stretch(I_AIHS(:,:,RGB_indexes),th_MSrgb));
