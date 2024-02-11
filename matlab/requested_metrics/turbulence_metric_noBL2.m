% function assumes that the Lapplacian has already been taken of both img_0
% and img_1
function [m] = turbulence_metric_noBL2(img_0, img_1)
    % img_0 is the reference image
    % img_1 is the simulated image

    m = 0;
    
    mean_img_0 = mean(img_0(:));
    mean_img_1 = mean(img_1(:));
    
    % step 1: take the FFT2 of the image to transform into the 2-D
    % frequency domain
    img_0_fft = fftshift(fft2(img_0 - mean_img_0)/numel(img_0));
    img_1_fft = fftshift(fft2(img_1 - mean_img_1)/numel(img_1));
    
    % step 2: Auto correlation
    % K_00 = conv2(img_0_fft, conj(img_0_fft(end:-1:1, end:-1:1)), 'same');
    K_00 = img_0_fft .* conj(img_0_fft(end:-1:1, end:-1:1));
    
    % step 3: Cross correlation
    % K_01 = conv2(img_0_fft, conj(img_1_fft(end:-1:1, end:-1:1)), 'same');
    K_01 = img_0_fft .*conj(img_1_fft(end:-1:1, end:-1:1));
    
    % step 4: calculate the sums of the correleations
    tmp_K = (K_01./K_00);
    tmp_K_mean = mean(tmp_K(:));

    sum_K_00 = mean((K_00(:)));
    sum_K_01 = mean((K_01(:)));
    
    % step 5: calculate the final metric
    m = 1 - abs(1 - abs(sum_K_01)/abs(sum_K_00));

    % m = 1 - abs(tmp_K_mean);
    
end
