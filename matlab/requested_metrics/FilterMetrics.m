function [metricsF] = FilterMetrics(filterType, dirBase, dirSharp, basefileN, ImgNames, sigma)

metricsF = zeros(length(ImgNames),length(sigma));

% Read in baseline image
pathB = fullfile(dirBase, basefileN);
ImageB = imread(pathB);
ImageB = ImageB(:,:,2);  % Use green channel

% Read in images
for i = 1:length(ImgNames)
    pathF = fullfile(dirSharp, ImgNames{i});
    Image = imread(pathF);
    Image = Image(:,:,2);  % Use green channel
    [M,N] = size(Image);

    % Metrics
    switch(filterType)
%     case('FFT of Diff Laplacian')       
%         k = fspecial('laplacian');
%         ImageLap = imfilter(double(Image),k,'symmetric');
%         ImageLapB = imfilter(double(ImageB),k,'symmetric');
%         fftB=fft2(double(ImageLap-ImageLapB),M,N);
%         resultF = abs(fftB);
%         metricsF(i) = sum(sum(resultF))/(M*N);
%         %imagesc(ImageLap), colormap('gray')
%         %imagesc(ImageLapB), colormap('gray')

    case('Diff of FFT Laplacian')
        k = fspecial('laplacian');
        ImageLap = imfilter(double(Image),k,'symmetric');
        ImageLapB = imfilter(double(ImageB),k,'symmetric');
        fftI=fft2(double(ImageLap),M,N);
        fftB=fft2(double(ImageLapB),M,N);
        resultF = abs(fftI-fftB);
        metricsF(i) = sum(sum(resultF))/(M*N);

%     case('FFT of Diff LoGs')    
%         for j = 1:length(sigma)
%             EdgeImg = edge(uint8(Image),"log",[],sigma(j)); % [EdgeImg,threshOut]
%             EdgeImgB = edge(uint8(ImageB),"log",[],sigma(j)); % [EdgeImg,threshOut]
%             fftB=fft2(double(EdgeImg-EdgeImgB),M,N);
%             %fftBshift = fftshift(fftB);
%             resultF = abs(fftB);
%             metricsF(i,j) = sum(sum(resultF))/(M*N);
%         end

    case('Diff of FFT LoGs')
        for j = 1:length(sigma)
            EdgeImg = edge(uint8(Image),"log",[],sigma(j)); 
            EdgeImgB = edge(uint8(ImageB),"log",[],sigma(j)); 
            fftI=fft2(double(EdgeImg),M,N);
            fftB=fft2(double(EdgeImgB),M,N);
            resultF = abs(fftI-fftB);
            metricsF(i,j) = sum(sum(resultF))/(M*N);
        end
%     case('Absolute Difference')
%         resultF = abs(Image - ImageB);
%         metricsF(i) = sum(sum(resultF))/(M*N);
    case('Absolute FFT')
        fftI = fft2(double(Image),M,N);
        fftB = fft2(double(ImageB),M,N);
        resultF = abs(fftI - fftB);
        metricsF(i) = sum(sum(resultF))/(M*N);
    end
end

end