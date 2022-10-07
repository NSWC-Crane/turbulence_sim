format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

commandwindow;

%% create a 1-D version of the checkerboard

block_w = 16;

img_min = 50;
img_max = 220;

data = repmat(cat(2, img_min*ones(1,block_w), img_max*ones(1,block_w)), 1, 8);


%% warp the image

tmp_psf = rand(1,8);
psf = resample(tmp_psf, 1,1);
psf = psf/(sum(psf));

data_warp1 = conv(data, psf(end:-1:1), 'same');

tmp_psf = rand(1,8);
psf = resample(tmp_psf, 1,1);
psf = psf/(sum(psf));

data_warp2 = conv(data, psf(end:-1:1), 'same');


figure
hold on
plot(data, 'b');
plot(data_warp1, 'r');
plot(data_warp2, 'g');

%% laplacian
lk = [1, -2, 1];

lap_data = conv(data, lk, 'same');

lap_data_w1 = conv(data_warp1, lk, 'same');
lap_data_w2 = conv(data_warp2, lk, 'same');


figure
hold on
plot(lap_data, 'b');
plot(lap_data_w1, 'r');
plot(lap_data_w2, 'g');

%% FFT


d_fft = fftshift(fft(data)/numel(data));
dw1_fft = fftshift(fft(data_warp1)/numel(data_warp1));
dw2_fft = fftshift(fft(data_warp2)/numel(data_warp2));

diff_fft_dw01 = d_fft - dw1_fft;
diff_fft_dw02 = d_fft - dw2_fft;


ld_fft = fftshift(fft(lap_data)/numel(lap_data));

ld_1_fft = fftshift(fft(lap_data_w1)/numel(lap_data_w1));

ld_2_fft = fftshift(fft(lap_data_w2)/numel(lap_data_w2));


diff_fft_01 = ld_1_fft - ld_fft;

diff_fft_02 = ld_2_fft - ld_fft;


figure;
hold on
plot((abs(d_fft)), 'b')
plot((abs(dw1_fft)), 'r')
plot((abs(dw2_fft)), 'g')

figure;
hold on
plot((abs(ld_fft)), 'b')
plot((abs(ld_1_fft)), 'r')
plot((abs(ld_2_fft)), 'g')


figure;
hold on
plot((abs(diff_fft_dw01)), 'r')
plot((abs(diff_fft_dw02)), 'g')

figure;
hold on
plot((abs(diff_fft_01)), 'r')
plot((abs(diff_fft_02)), 'g')


%% correlate

cv_diff_dw01 = conv(diff_fft_dw01, diff_fft_dw01(end:-1:1), 'same');
cv_diff_dw12 = conv(diff_fft_dw01, diff_fft_dw02(end:-1:1), 'same');


cv_diff_f01 = conv(diff_fft_01, diff_fft_01(end:-1:1), 'same');
cv_diff_f12 = conv(diff_fft_01, diff_fft_02(end:-1:1), 'same');


figure;
hold on
plot(abs(cv_diff_dw01),'b')
plot(abs(cv_diff_dw12),'r')
% plot(real(cv_diff_f01),'r')
% plot(imag(cv_diff_f01),'g')


figure;
hold on
plot(abs(cv_diff_f01),'b')
plot(abs(cv_diff_f12),'r')
% plot(real(cv_diff_f12),'r')
% plot(imag(cv_diff_f12),'g')

s_01 = sum(abs(cv_diff_f01(:)));
s_02 = sum(abs(cv_diff_f12(:)));

r012 = s_02/s_01

r012m = max(abs(cv_diff_f12(:)))/max(abs(cv_diff_f01(:)))

abs(1-r012m)


s_dw01 = sum(abs(cv_diff_dw01));
s_dw12 = sum(abs(cv_diff_dw12));

r_dw12 = s_dw12/s_dw01



