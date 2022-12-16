clearvars
clc

platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% % Range 600, Zoom 3000
% % Real:  C:\Data\JSSAP\sharpest\z3000\0600\image_z02996_f47005_e15004_i00.png
% % Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c3e14_N1.png
% % Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c5e15_N1.png
% 
ref_image1_Files = data_root + "sharpest\z3000\0600\image_z02996_f47005_e15004_i00.png";
test_images1_Files = [data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c3e14_N1.png",...
    data_root + "modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c5e15_N1.png"];

rngstr = "600";
zmstr = "3000";
% Read in real image
refImg = double(imread(ref_image1_Files));
refImg= refImg(:,:,2);  % Only green channel
% Subtract Mean of Image
refImg= refImg - mean(refImg,'all');
% Remove 5 pixels off borders
%refImg = refImg(31:end-30, 31:end-30); % (6:end-5, 6:end-5);
refImg = refImg(6:end-5, 6:end-5);

% Read in max metric test image
testImgMax = double(imread(test_images1_Files(1)));
% Subtract Mean of Image
testImgMax= testImgMax - mean(testImgMax,'all');
% Remove 5 pixels off borders
%testImgMax = testImgMax(31:end-30, 31:end-30); % (6:end-5, 6:end-5);
testImgMax = testImgMax(6:end-5, 6:end-5);
scoreMax = CW_SSIM(refImg,testImgMax);



% Read in closest to real test image
testImgClose = double(imread(test_images1_Files(2)));
% Subtract Mean of Image
testImgClose= testImgClose - mean(testImgClose,'all');
% Remove 5 pixels off borders
%testImgClose = testImgClose(31:end-30, 31:end-30); % (6:end-5, 6:end-5);
testImgClose = testImgClose(6:end-5, 6:end-5);
testImgCl_gauss = imgaussfilt(testImgClose,.69);

scoreClose = CW_SSIM(refImg, testImgClose);
scoreCloseG = CW_SSIM(refImg, testImgCl_gauss);

[ac1, ccMax] = xcorrImages(refImg, testImgMax);
[ac2, ccClose] = xcorrImages(refImg, testImgClose);




% m1 = turbulence_metric_noBL_testVL(refImg, testImgMax,"mm", rngstr, zmstr);
% 
% m2 = turbulence_metric_noBL_testVL(refImg, testImgClose,"c", rngstr, zmstr);
% 
% m2g = turbulence_metric_noBL_testVL(refImg, testImgCl_gauss,"c", rngstr, zmstr);