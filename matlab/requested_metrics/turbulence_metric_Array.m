
function [m] = turbulence_metric_Array(ImgArray_0, ImgArray_1)
    % img_0 is the reference image
    % img_1 is the simulated image
    % Used to process an array of windows created from the original image
    % All windows must be the same size image
    
    [~,~,dp] = size(ImgArray_0);

%     % Preprocessing:  Eliminate below for testing windows
%     numElems = numel(ImgArray_0(:,:,1));
%     Mean_Img_0 = mean(ImgArray_0,[1 2]);
%     % test mean2?
%     Mean_Img_1 = mean(ImgArray_1,[1 2]);
%     Nom_Img_0  = ImgArray_0 - Mean_Img_0;
%     Nom_Img_1  = ImgArray_1 - Mean_Img_1;
% 
%     % step 1: take the FFT2 of the images to transform into the 2-D
%     % frequency domain
%     Img_0_FFT = fftshift(fft2(Nom_Img_0)./numElems);
%     Img_1_FFT = fftshift(fft2(Nom_Img_1)./numElems);

    % step 1: take the FFT2 of the images to transform into the 2-D
    % frequency domain
    numElems = numel(ImgArray_0(:,:,1));
    Img_0_FFT = fftshift(fft2(ImgArray_0)./numElems);
    Img_1_FFT = fftshift(fft2(ImgArray_1)./numElems);
    
    % step 2: Auto correlation
    % First, get conjugate
    conjImg0 = conj(Img_0_FFT(end:-1:1, end:-1:1,:));
    % Then auto correlation
    for i = 1:dp
        K_00(:,:,i) = conv2(Img_0_FFT(:,:,i),conjImg0(:,:,i),'same');
    end
    
    % step 3: Cross correlation
    % First, get conjugate
    conjImg1 = conj(Img_1_FFT(end:-1:1, end:-1:1,:));
    % Then cross correlation
    for i = 1:dp
        K_01(:,:,i) = conv2(Img_0_FFT(:,:,i),conjImg1(:,:,i),'same');
    end
    
    % step 4: calculate the sums of the correleations
    Sum_K_00 = sum(abs(K_00),[1,2]);
    Sum_K_01 = sum(abs(K_01),[1,2]);
  
    % step 5: calculate the final metric
    m = 1-abs(1-(Sum_K_01(:,:)./Sum_K_00(:,:)));

        
end
