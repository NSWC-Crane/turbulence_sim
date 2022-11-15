% function assumes that the Lapplacian has already been taken of both img_0
% and img_1
function [m] = turbulence_metric_noBL(img_0, img_1)

    m = 0;
    
    % step 1: take the FFT2 of the image to transform into the 2-D
    % frequency domain
    img_0_fft = fftshift(fft2(img_0)/numel(img_0));
    img_1_fft = fftshift(fft2(img_1)/numel(img_1));
    
    % step 2: Auto correlation
    K_00 = conv2(img_0_fft, conj(img_0_fft(end:-1:1, end:-1:1)), 'same');
    
    % step 3: Cross correlation
    K_01 = conv2(img_0_fft, conj(img_1_fft(end:-1:1, end:-1:1)), 'same');
    
    % step 4: calculate the sums of the correleations
    sum_K_00 = sum(abs(K_00(:)));
    sum_K_01 = sum(abs(K_01(:)));
    
    % step 5: calculate the final metric
    m = 1 - abs(1 - (sum_K_01/sum_K_00));
    
end
