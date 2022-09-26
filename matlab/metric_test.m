format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[scriptpath,  filename, ext] = fileparts(full_path);
plot_count = 1;
line_width = 1.0;

commandwindow;

%%
num_images = 10;
file_path = 'D:\data\turbulence\sharpest\z2000\';

% base = double(imread(strcat(file_path, 'baseline_z2000_r0600.png')));
base = double(imread(strcat(file_path, 'baseline_z2000_r0600_i00.png')));
base = base(:,:,2)/255;

num_pix = numel(base);


img = cell(num_images,1);

figure(plot_count)
set(gcf,'position',([100,100,1500,900]),'color','w')
for idx=1:num_images
    %z5000
%     tmp_img = double(imread(strcat(file_path, '0850\image_z05000_f48030_e02958_i', num2str(idx-1,'%02d'),'.png')));
%     tmp_img = double(imread(strcat(file_path, '1000\image_z05000_f48069_e02842_i', num2str(idx-1,'%02d'),'.png')));
    
    %z2000
    tmp_img = double(imread(strcat(file_path, '0600\image_z01998_f46229_e14987_i', num2str(idx-1,'%02d'),'.png')));
%     tmp_img = double(imread(strcat(file_path, '0700\image_z01998_f46270_e06064_i', num2str(idx-1,'%02d'),'.png')));
%     tmp_img = double(imread(strcat(file_path, '0800\image_z01999_f46270_e07221_i', num2str(idx-1,'%02d'),'.png')));
%     tmp_img = double(imread(strcat(file_path, '0900\image_z01999_f46249_e03553_i', num2str(idx-1,'%02d'),'.png')));
%     tmp_img = double(imread(strcat(file_path, '1000\image_z01999_f46264_e02776_i', num2str(idx-1,'%02d'),'.png')));
    
    img{idx} = tmp_img(:,:,2)/255;
    
    subplot(4, 5, idx)
    imagesc(img{idx});
    colormap(gray(256));
    axis off;
end
plot_count = plot_count + 1;

%%
d_img = cell(num_images, num_images);
f_img = cell(num_images, num_images);
fd_img = cell(num_images, num_images);
db_img = cell(num_images, 1);
fb_img = cell(num_images, 1);

d_sum = zeros(num_images, num_images);
f_sum = zeros(num_images, num_images);
fd_sum = zeros(num_images, num_images);
db_sum = zeros(num_images, 1);
fb_sum = zeros(num_images, 1);

for idx=1:num_images
    for jdx=idx:num_images
        
        d_img{idx,jdx} = (img{idx} - img{jdx})/(numel(img{idx}));
        d_sum(idx,jdx) = sum(sum(abs(d_img{idx,jdx})))/num_pix;
        
        f_img{idx,jdx} = fft2(img{idx})/numel(img{idx});     
        f_sum(idx,jdx) = sum(sum(abs(f_img{idx,jdx})))/num_pix;
        
        fd_img{idx,jdx} = fft2((img{idx} - img{jdx}))/(numel(img{idx}));
        fd_sum(idx,jdx) = sum(sum(abs(fd_img{idx,jdx})))/num_pix;

    end
    
    db_img{idx,1} = (abs(img{idx} - base))/numel(img{idx});
    db_sum(idx,1) = sum(sum((db_img{idx,1})))/num_pix;    
    
    fb_img{idx,1} = (abs(fft2(img{idx}) - (fft2(base))))/numel(img{idx});
    fb_sum(idx,1) = sum(sum((fb_img{idx,1})))/num_pix;    
    
end

%% plotting

figure(plot_count)
set(gcf,'position',([100,100,1500,900]),'color','w')
for idx=1:num_images
    for jdx=idx:num_images
        
        subplot(num_images, num_images, (idx-1)*num_images + jdx)
        imagesc(d_img{idx,jdx});
        colormap(gray(256));
        axis off;
    end
end
plot_count = plot_count + 1;

figure(plot_count)
set(gcf,'position',([100,100,1500,900]),'color','w')
for idx=1:num_images
    for jdx=idx:num_images
        
        subplot(num_images, num_images, (idx-1)*num_images + jdx)
        surf(fftshift(abs(f_img{idx,jdx})));
        colormap(jet(256));
        shading interp
    end
end
plot_count = plot_count + 1;

figure(plot_count)
set(gcf,'position',([100,100,1500,900]),'color','w')
for idx=1:num_images
    for jdx=idx:num_images
        
        subplot(num_images, num_images, (idx-1)*num_images + jdx)
        surf(fftshift(abs(fd_img{idx,jdx})));
        colormap(jet(256));
        shading interp
    end
end
plot_count = plot_count + 1;

figure(plot_count)
set(gcf,'position',([100,100,1500,900]),'color','w')
for idx=1:num_images
%     for jdx=idx:num_images
        
        subplot(num_images, 1, idx)
        surf(fftshift(abs(fb_img{idx,1})));
        colormap(jet(256));
        shading interp
%     end
end
plot_count = plot_count + 1;


fprintf('d sum\n');
disp(d_sum)

fprintf('f sum\n');
disp(f_sum)

fprintf('fd sum\n');
disp(fd_sum)

fprintf('fb sum\n');
disp(fb_sum)

mean_d_sum = sum(d_sum(:))/nnz(d_sum);
fprintf('\nmean d sum: %4.8f\n', mean_d_sum);

mean_db_sum = sum(db_sum(:))/nnz(db_sum);
fprintf('\nmean db sum: %4.8f\n', mean_db_sum);

mean_f_sum = sum(f_sum(:))/nnz(f_sum);
fprintf('mean f sum: %4.8f\n', mean_f_sum);

mean_fd_sum = sum(fd_sum(:))/nnz(fd_sum);
fprintf('mean fd sum: %4.8f\n', mean_fd_sum);

mean_fb_sum = sum(fb_sum(:))/nnz(fb_sum);
fprintf('mean fb sum: %4.8f\n', mean_fb_sum);


%%
% figure(plot_count)
% set(gcf,'position',([100,100,1500,900]),'color','w')
% 
% subplot(1, 2, 1)
% surf(fftshift(abs(fd_img{1,2})));
% colormap(jet(256));
% shading interp
%         
% subplot(1, 2, 2)
% surf(fftshift((fb_img{1})));
% colormap(jet(256));
% shading interp
% 
% plot_count = plot_count + 1;

