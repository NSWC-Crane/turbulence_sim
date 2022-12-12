% function assumes that the Lapplacian has already been taken of both img_0
% and img_1
function [m] = turbulence_metric_noBL(img_0, img_1, mtype, rngstr, zmstr)
    % img_0 is the reference image
    % img_1 is the simulated image
    %mtype:  'c' is Closest, 'mm' is Max Metric

    m = 0;
    
    % step 1: take the FFT2 of the image to transform into the 2-D
    % frequency domain
    img_0_fft = fftshift(fft2(img_0)/numel(img_0));
    img_1_fft = fftshift(fft2(img_1)/numel(img_1));

    if mtype == "c"
        strM = "Closest Image";
    elseif mtype == "mm"
            strM = "Max Metric Image";
    else
        fprintf("Warning:  Improper mtype");
    end

    figure()
    imagesc(abs(fftshift(img_0_fft)))
    title("Real Image FFT:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    imagesc(abs(fftshift(img_1_fft)))
    title(strM + " FFT:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    imagesc(abs(fftshift(log2(img_0_fft))))
    title("Real Image FFT:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    imagesc(abs(fftshift(log2(img_1_fft))))
    title(strM + " FFT:  Rng " + rngstr + ", Zm " + zmstr)

    figure()
    plot(abs(img_0_fft))
    title("Real Image FFT:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    plot(abs(img_1_fft))
    title(strM + " FFT:  Rng " + rngstr + ", Zm " + zmstr)

    figure()
    mesh(abs(img_0_fft))
    title("Real Image FFT:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    mesh(abs(img_1_fft))
    title(strM + " FFT:  Rng " + rngstr + ", Zm " + zmstr)
    

    
    % step 2: Auto correlation
    K_00 = conv2(img_0_fft, conj(img_0_fft(end:-1:1, end:-1:1)), 'same');
    
    % step 3: Cross correlation
    K_01 = conv2(img_0_fft, conj(img_1_fft(end:-1:1, end:-1:1)), 'same');

    figure()
    mesh(abs(K_00))
    title("K_00 Convolution:  Rng " + rngstr + ", Zm " + zmstr)
    figure()
    mesh(abs(K_01))
    title("K_01 Convolution " + strM + ":  Rng " + rngstr + ", Zm " + zmstr)
    
    % step 4: calculate the sums of the correleations
    sum_K_00 = sum(abs(K_00(:)));
    sum_K_01 = sum(abs(K_01(:)));
    
    % step 5: calculate the final metric
    m = 1 - abs(1 - (sum_K_01/sum_K_00));
    
end
