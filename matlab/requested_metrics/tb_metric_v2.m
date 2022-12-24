% function assumes that the Lapplacian has already been taken of both img_0
% and img_1
function [m] = tb_metric_v2(img_0, img_1)
    % img_0 is the reference image
    % img_1 is the simulated image

    m = 0;
    c1 = 0.01;
    
    mean_img_0 = mean(img_0(:));
    mean_img_1 = mean(img_1(:));
    
    % step 1: take the FFT2 of the image to transform into the 2-D
    % frequency domain
    img_0_fft = fftshift(fft2(img_0 - mean_img_0)/numel(img_0));
    img_1_fft = fftshift(fft2(img_1 - mean_img_1)/numel(img_1));
    
    % step 2: Auto correlation
    K_00 = conv2(img_0_fft, conj(img_0_fft(end:-1:1, end:-1:1)), 'same');
    
    % step 3: Cross correlation
    K_01 = conv2(img_0_fft, conj(img_1_fft(end:-1:1, end:-1:1)), 'same');
    
    V_01 = abs(img_0_fft).^2 + abs(img_1_fft).^2;
    
    tb_map = (2*abs(K_01) + c1)./(V_01 + c1);
    
    % step 4: calculate the sums of the correleations
%     sum_K_00 = sum(abs(K_00(:)));
%     sum_K_01 = sum(abs(K_01(:)));
    
    % step 5: calculate the final metric
%     m = 1 - abs(1 - (sum_K_01/sum_K_00));
    

    m = sum(tb_map(:));


end
