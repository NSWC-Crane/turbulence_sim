clearvars
clc

%% Range 600, Zoom 3000 
ref_image1_Files = "C:\Data\JSSAP\sharpest\z3000\0600\image_z02996_f47005_e15004_i00.png";
test_images1_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c3e14_N1.png",...
    "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c5e15_N1.png"];

rngstr = "600";
zmstr = "3000";
% Read in real image
refImg = double(imread(ref_image1_Files));
refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% Remove 5 pixels off borders
refImg = refImg(6:end-5, 6:end-5);

% Read in max metric test image
testImgMax = double(imread(test_images1_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% Remove 5 pixels off borders
testImgMax = testImgMax(6:end-5, 6:end-5);

% Read in closest to real test image
testImgClose = double(imread(test_images1_Files(2)));
% % Subtract Mean of Image
% testImgClose= testImgClose - mean(testImgClose,'all');
% Remove 5 pixels off borders
testImgClose = testImgClose(6:end-5, 6:end-5);
testImgCl_gauss = imgaussfilt(testImgClose,.69);

% cwssim = cwssim_index(img1, img2,6,16,0,0);
level = 1;
or = 1; % 10
guardb = 0;
K = 0;

cwssimMax = cwssim_index(refImg, refImg*.5, level, or, guardb, K);
cwssimCloseN3 = cwssim_index(refImg, testImgClose+(randn(246,246)*3), level, or, guardb, K);
cwssimCloseN2 = cwssim_index(refImg, testImgClose+(randn(246,246)*2), level, or, guardb, K);
cwssimClose = cwssim_index(refImg, testImgClose, level, or, guardb, K);
cwssimCloseG = cwssim_index(refImg, testImgCl_gauss, level, or, guardb, K);


%% Range 600, Zoom 2500
ref_image2_Files = "C:\Data\JSSAP\sharpest\z2500\0600\image_z02498_f46581_e15004_i00.png";
test_images2_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z2500_c1e13_N1.png",...
    "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z2500_c1e14_N1.png"];

rngstr = "600";
zmstr = "2500";
% Read in real image
refImg = double(imread(ref_image2_Files));
refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% Remove 5 pixels off borders
refImg = refImg(6:end-5, 6:end-5);

% Read in max metric test image
testImgMax = double(imread(test_images2_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% Remove 5 pixels off borders
testImgMax = testImgMax(6:end-5, 6:end-5);

% Read in closest to real test image
testImgClose = double(imread(test_images2_Files(2)));
% Remove 5 pixels off borders
testImgClose = testImgClose(6:end-5, 6:end-5);
testImgCl_gauss = imgaussfilt(testImgClose,.69);

cwssimMax2 = cwssim_index(refImg, testImgMax, level, or, guardb, K);
cwssimClose2 = cwssim_index(refImg, testImgClose, level, or, guardb, K);
cwssimCloseG2 = cwssim_index(refImg, testImgCl_gauss, level, or, guardb, K);

%% Range 650, Zoom 2500
% Real:  C:\Data\JSSAP\sharpest\z2500\0650\image_z02497_f46620_e11187_i00.png
% Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e13_N1.png
% Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e15_N1.png

ref_image3_Files = "C:\Data\JSSAP\sharpest\z2500\0650\image_z02497_f46620_e11187_i00.png";
test_images3_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e13_N1.png",...
    "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e15_N1.png"];

rngstr = "650";
zmstr = "2500";
% Read in real image
refImg = double(imread(ref_image3_Files));
refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% Remove 5 pixels off borders
refImg = refImg(6:end-5, 6:end-5);

% Read in max metric test image
testImgMax = double(imread(test_images3_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% Remove 5 pixels off borders
testImgMax = testImgMax(6:end-5, 6:end-5);

% Read in closest to real test image
testImgClose = double(imread(test_images3_Files(2)));
% Remove 5 pixels off borders
testImgClose = testImgClose(6:end-5, 6:end-5);
testImgCl_gauss = imgaussfilt(testImgClose,.69);

cwssimMax3 = cwssim_index(refImg, testImgMax, level, or, guardb, K);
cwssimClose3 = cwssim_index(refImg, testImgClose, level, or, guardb, K);
cwssimCloseG3 = cwssim_index(refImg, testImgCl_gauss, level, or, guardb, K);


