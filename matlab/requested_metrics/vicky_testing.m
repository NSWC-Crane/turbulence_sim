% Run metric on selected images

clearvars
clc

file_RealImg = "C:\Data\JSSAP\modifiedBaselines\NewSimulations\Reals\Real_600_z4000_N1.png";
file_RealImg2 = "C:\Data\JSSAP\sharpest\z4000\0600\image_z03995_f47695_e15004_i07.png";
file_SimImg = "C:\Data\JSSAP\modifiedBaselines\NewSimulations\ByVaryingCn2\NewSim_r600_z4000_c5e15_N1.png";

% Read in images and clip border
borderSize = 5;
RealImg = double(imread(file_RealImg2));
[img_h,img_w,img_d] = size(RealImg);
if img_d > 1
    RealImg = RealImg(:,:,2);
end
RealImg = RealImg(borderSize + 1:img_h-borderSize,borderSize + 1:img_w-borderSize);

SimImg = double(imread(file_SimImg));
[img_h,img_w] = size(SimImg);
SimImg = SimImg(borderSize + 1:img_h-borderSize,borderSize + 1:img_w-borderSize);

[m, K_01, img_0_fft, img_1_fft] = turbulence_metricVL(RealImg, SimImg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reals vary
clear
clc

rdir = "C:\Data\JSSAP\sharpest\z5000\0650\";
sfile = "C:\Data\JSSAP\modifiedBaselines\NewSimulations\SimReal2\NewSim_r650_z5000_N1.png";

% get metrics for all reals in rdir as compared to simulated image sfile.
image_ext = '*.png';
rlisting = dir(strcat(rdir, '/', image_ext));

% Read in images and clip border
borderSize = 5;
SimImg = double(imread(sfile));
[img_h,img_w] = size(SimImg);
SimImg = SimImg(borderSize + 1:img_h-borderSize,borderSize + 1:img_w-borderSize);

m = ones(length(rlisting),1);

for idx = 1:length(rlisting)
    % Read in images and clip border
    rname = fullfile(rdir, '/', rlisting(idx).name);
    RealImg = double(imread(rname));
    [img_h,img_w,img_d] = size(RealImg);
    if img_d > 1
        RealImg = RealImg(:,:,2);
    end
    RealImg = RealImg(borderSize + 1:img_h-borderSize,borderSize + 1:img_w-borderSize);
    
    %[m(idx), K_01, img_0_fft, img_1_fft] = turbulence_metricVL(RealImg, SimImg);
    [m(idx), ~, ~, ~] = turbulence_metricVL(RealImg, SimImg);
end

% Plot metrics x-axis: image number, y-axis: metric m
xstr = ["i00","i01","i02","i03","i04","i05","i06","i07","i08","i09","i10",...
         "i11","i12","i13","i14","i15","i16","i17","i18","i19"];
x = 0:19;
figure()
plot(x, m, '-b.')
grid on
ylim([0.3,1.0])
xlim([0,19])
title("Range 650 Zoom 5000: All reals compared to one simulated real image")
xlabel("Image label")
ylabel("Metric")

