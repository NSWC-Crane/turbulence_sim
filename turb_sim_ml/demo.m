format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[scriptpath,  filename, ext] = fileparts(full_path);

%%

img_in  = im2double(imread(strcat(scriptpath, '/../data/checker_board_32x32.png')));
if size(img_in,3)~=1
    img  = rgb2gray(img_in);
end

% crop instead of resizing
%img = img(1:256, 1:256);
%img = imresize(img,[256,256]);

params.t_params.D = 0.095;
params.t_params.L = 800;
params.t_params.d = 1.38;
params.t_params.Cn2 = 1e-14;
params.t_params.lambda = 0.525e-6;

opts.frames = 1;

[temp, ph, outPSF] = sim_fun(img, params, opts);

out_stack = uint8(256*temp);

return;

%% plot some data <- not actually run, but used for cut and paste while debugging

% this one is for blur and then motion compensate

figure(1)
set(gcf,'position',([50,50,1200,500]),'color','w')

subplot(1,2,1)
box on
hold on
axis off
imagesc(img_blur(:,:,k))
xlim([0, size(img_blur(:,:,k), 2)])
ylim([0, size(img_blur(:,:,k), 1)])
title('Image Blur', 'fontweight','bold','FontSize',11);

subplot(1,2,2)
box on
hold on
axis off
imagesc(img_out(:,:,k))
xlim([0, size(img_out(:,:,k), 2)])
ylim([0, size(img_out(:,:,k), 1)])
title('Image Blur & Motion Compensate', 'fontweight','bold','FontSize',11);

colormap(gray(256))

%% this one is for motion compensate and then blur

figure(1)
set(gcf,'position',([50,50,1200,500]),'color','w')

subplot(1,2,1)
box on
hold on
axis off
imagesc(img_mc(:,:,k))
xlim([0, size(img_mc(:,:,k), 2)])
ylim([0, size(img_mc(:,:,k), 1)])
title('Motion Compensate', 'fontweight','bold','FontSize',11);

subplot(1,2,2)
box on
hold on
axis off
imagesc(img_out(:,:,k))
xlim([0, size(img_out(:,:,k), 2)])
ylim([0, size(img_out(:,:,k), 1)])
title('Motion Compensate & Image Blur', 'fontweight','bold','FontSize',11);

colormap(gray(256))

