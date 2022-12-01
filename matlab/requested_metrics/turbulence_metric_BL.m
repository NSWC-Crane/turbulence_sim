% function assumes that the Lapplacian has already been taken of both
% img_0, img_1 and img_b
function [m] = turbulence_metric_BL(img_0, img_1, img_b)
    % img_0 is the reference image
    % img_1 is the simulated image

    m = 0;
    
    % step 1: take the FFT2 of the image to transform into the 2-D
    % frequency domain
    img_0_fft = fftshift(fft2(img_0)/numel(img_0));
    img_1_fft = fftshift(fft2(img_1)/numel(img_1));
    img_b_fft = fftshift(fft2(img_b)/numel(img_b));
    
    % step 2: take the difference between the image and the baseline
    diff_img_0b = img_0_fft - img_b_fft;
    diff_img_1b = img_1_fft - img_b_fft;
    
    % step 2: Auto correlation
    K_00 = conv2(diff_img_0b, conj(diff_img_0b(end:-1:1, end:-1:1)), 'same');
    
    % step 3: Cross correlation
    K_01 = conv2(diff_img_0b, conj(diff_img_1b(end:-1:1, end:-1:1)), 'same');
    
    % step 4: calculate the sums of the correleations
    sum_K_00 = sum(abs(K_00(:)));
    sum_K_01 = sum(abs(K_01(:)));
    
    % step 5: calculate the final metric
    m = 1 - abs(1 - (sum_K_01/sum_K_00));
    
end
