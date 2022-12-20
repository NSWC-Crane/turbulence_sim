% Take images that seem to be very different
% Why are the metrics close?

% 1. Read in images
% 2. Use only green channel
% 3. Subtract mean(image)
% 4. Remove 5 pixels from each edge to simulate "One Patch"
% 5. Call Metric Function
% 6. Plot FFTs (both dimensions)
% 7. Plots sums of each frequency
% 8. Compare sums

% Ref:  Real Image
% Test Images: Max metric, Closest metric

clearvars
clc

% % Range 600, Zoom 3000
% % Real:  C:\Data\JSSAP\sharpest\z3000\0600\image_z02996_f47005_e15004_i00.png
% % Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c3e14_N1.png
% % Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c5e15_N1.png
% 
% Setup data directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end


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
refImg = refImg(5:end-5, 5:end-5);

% Read in max metric test image
testImgMax = double(imread(test_images1_Files(1)));
% Subtract Mean of Image
testImgMax= testImgMax - mean(testImgMax,'all');
% Remove 5 pixels off borders
testImgMax = testImgMax(5:end-5, 5:end-5);

% Read in closest to real test image
testImgClose = double(imread(test_images1_Files(2)));
% Subtract Mean of Image
testImgClose= testImgClose - mean(testImgClose,'all');
% Remove 5 pixels off borders
testImgClose = testImgClose(5:end-5, 5:end-5);
testImgCl_gauss = imgaussfilt(testImgClose,.69);

m1 = turbulence_metric_noBL_testVL(refImg, testImgMax,"mm", rngstr, zmstr);

m2 = turbulence_metric_noBL_testVL(refImg, testImgClose,"c", rngstr, zmstr);

m2g = turbulence_metric_noBL_testVL(refImg, testImgCl_gauss,"c", rngstr, zmstr);

% % Range 600, Zoom 2500
% % Real:  C:\Data\JSSAP\sharpest\z2500\0600\image_z02498_f46581_e15004_i00.png
% % Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c1e13_N1.png
% % Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z3000_c1e14_N1.png
% 
% ref_image2_Files = "C:\Data\JSSAP\sharpest\z2500\0600\image_z02498_f46581_e15004_i00.png";
% test_images2_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z2500_c1e13_N1.png",...
%     "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z2500_c1e14_N1.png"];
% 
% rngstr = "600";
% zmstr = "2500";
% % Read in real image
% refImg = double(imread(ref_image2_Files));
% refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% % Remove 5 pixels off borders
% refImg = refImg(5:end-5, 5:end-5);
% 
% % Read in max metric test image
% testImgMax = double(imread(test_images2_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% % Blur max metric image
% % Remove 5 pixels off borders
% testImgMax = testImgMax(5:end-5, 5:end-5);
% 
% % Read in closest to real test image
% testImgClose = double(imread(test_images2_Files(2)));
% % figure()
% % imshow(uint8(testImgClose))
% % blurI = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(blurI))
% % Subtract Mean of Image
% testImgClose= testImgClose - mean(testImgClose,'all');
% % figure()
% % imshow(uint8(testImgClose))
% % Blur closest image
% %testImgClose = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(testImgClose))
% % Remove 5 pixels off borders
% testImgClose = testImgClose(5:end-5, 5:end-5);
% testImgCl_gauss = imgaussfilt(testImgClose,1.5);
% 
% m3 = turbulence_metric_noBL_testVL(refImg, testImgMax,"mm", rngstr, zmstr);
% 
% m4 = turbulence_metric_noBL_testVL(refImg, testImgClose,"c", rngstr, zmstr);
% m4g = turbulence_metric_noBL_testVL(refImg, testImgCl_gauss,"c", rngstr, zmstr);
% 
% % Range 650, Zoom 2500
% % Real:  C:\Data\JSSAP\sharpest\z2500\0650\image_z02497_f46620_e11187_i00.png
% % Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e13_N1.png
% % Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e15_N1.png
% 
% ref_image3_Files = "C:\Data\JSSAP\sharpest\z2500\0650\image_z02497_f46620_e11187_i00.png";
% test_images3_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e13_N1.png",...
%     "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z2500_c3e15_N1.png"];
% 
% rngstr = "650";
% zmstr = "2500";
% % Read in real image
% refImg = double(imread(ref_image3_Files));
% refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% % Remove 5 pixels off borders
% refImg = refImg(5:end-5, 5:end-5);
% 
% % Read in max metric test image
% testImgMax = double(imread(test_images3_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% % Blur max metric image
% % Remove 5 pixels off borders
% testImgMax = testImgMax(5:end-5, 5:end-5);
% 
% % Read in closest to real test image
% testImgClose = double(imread(test_images3_Files(2)));
% % figure()
% % imshow(uint8(testImgClose))
% % blurI = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(blurI))
% % Subtract Mean of Image
% testImgClose= testImgClose - mean(testImgClose,'all');
% % figure()
% % imshow(uint8(testImgClose))
% % Blur closest image
% %testImgClose = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(testImgClose))
% % Remove 5 pixels off borders
% testImgClose = testImgClose(5:end-5, 5:end-5);
% testImgCl_gauss = imgaussfilt(testImgClose,0.65);
% 
% m5 = turbulence_metric_noBL_testVL(refImg, testImgMax,"mm", rngstr, zmstr);
% 
% m6 = turbulence_metric_noBL_testVL(refImg, testImgClose,"c", rngstr, zmstr);
% m6g = turbulence_metric_noBL_testVL(refImg, testImgCl_gauss,"c", rngstr, zmstr);
% 
% 
% % Range 650, Zoom 3000
% % Real:  C:\Data\JSSAP\sharpest\z3000\0650\image_z03000_f47035_e09303_i00.png
% % Max metric:  C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z3000_c9e15_N1.png
% % Closest real: C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z3000_c1e15_N1.png
% 
% ref_image3_Files = "C:\Data\JSSAP\sharpest\z3000\0650\image_z03000_f47035_e09303_i00.png";
% test_images3_Files = ["C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z3000_c9e15_N1.png",...
%     "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r650_z3000_c1e15_N1.png"];
% 
% rngstr = "650";
% zmstr = "3000";
% % Read in real image
% refImg = double(imread(ref_image3_Files));
% refImg= refImg(:,:,2);  % Only green channel
% % Subtract Mean of Image
% refImg= refImg - mean(refImg,'all');
% % Remove 5 pixels off borders
% refImg = refImg(5:end-5, 5:end-5);
% 
% % Read in max metric test image
% testImgMax = double(imread(test_images3_Files(1)));
% % Subtract Mean of Image
% testImgMax= testImgMax - mean(testImgMax,'all');
% % Blur max metric image
% % Remove 5 pixels off borders
% testImgMax = testImgMax(5:end-5, 5:end-5);
% 
% % Read in closest to real test image
% testImgClose = double(imread(test_images3_Files(2)));
% % figure()
% % imshow(uint8(testImgClose))
% % blurI = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(blurI))
% % Subtract Mean of Image
% testImgClose= testImgClose - mean(testImgClose,'all');
% % figure()
% % imshow(uint8(testImgClose))
% % Blur closest image
% %testImgClose = imgaussfilt(testImgClose, 1);
% % figure()
% % imshow(uint8(testImgClose))
% % Remove 5 pixels off borders
% testImgClose = testImgClose(5:end-5, 5:end-5);
% testImgCl_gauss = imgaussfilt(testImgClose,0.6);
% 
% m7 = turbulence_metric_noBL_testVL(refImg, testImgMax,"mm", rngstr, zmstr);
% 
% m8 = turbulence_metric_noBL_testVL(refImg, testImgClose,"c", rngstr, zmstr);
% m8g = turbulence_metric_noBL_testVL(refImg, testImgCl_gauss,"c", rngstr, zmstr);
% 
% 
% 
% 
