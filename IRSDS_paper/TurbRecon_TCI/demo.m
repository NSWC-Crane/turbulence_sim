%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo code for Image Reconstruction through Atmospheric Turbulence
%
% Reference:
% Z. Mao, N. Chimitt, and S. H. Chan,
% "Image Reconstruction of Static and Dynamic Scenes through Anisoplanatic 
% Turbulence", IEEE Transactions on Computational Imaging, vol. 6, 
% pp. 1415-1428, Oct. 2020. 
% 
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2021
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

format long g
format compact
clc
close all
clearvars

addpath(genpath('./downloads'))
addpath(genpath('./utils'))

% set input path and load data
% dir_name = 'z3000_r800_cropped';
% input_dir = ['./data/' dir_name '/'];
dir_name = 'fp2';
output_dir = './color_blur_turb_test/';
input_dir = ['./color_blur_turb_test/' dir_name '/'];

%input_dir = './data/scene4/';
imgs = dir([input_dir '*.png']);
for i = 1:length(imgs)
    temp = imread([input_dir imgs(i).name]);
    if size(temp,3) == 3
        %display(temp)
        temp = rgb2gray(temp);
    end
    stack(:,:,i) = double(temp);
end

% load parameters
load ./utils/medium.mat

reg_parameters.id_2 = -1;
% display(reg_parameters);
% display(deblur_parameters);
% outname = ['test_' dir_name '_' num2str(reg_parameters.id_2,'%i')];
outname = dir_name;

% registration step
fprintf('===========================\n');
fprintf('TurbRecon_v1, Purdue University, 2020\n');
fprintf('===========================\n');

fprintf('Part 1: Image registration\n')
[luck_out, reg_out, reg_stack] = registration_main(stack, reg_parameters);
fprintf('\n');

figs2 = figure;
subplot(121);
imshow(luck_out); title('luck frame');
subplot(122);
imshow(reg_out); title('reg frame');
figname = [output_dir outname '_luckvreg.png'];
exportgraphics(figs2, figname);
imwrite(uint8(luck_out*255),[output_dir outname '_luck.png']);
imwrite(uint8(reg_out*255),[output_dir outname '_reg.png']);

% blind deblurring step
fprintf('Part 2: Blind deconvolution \n')
[final_out, hnew] = deblur_main(luck_out,deblur_parameters);

fprintf('\n');
fprintf('Done. Display results.\n');

figs = figure;
subplot(121);
imshow(stack(:,:,end/2), []); title('an input frame');
subplot(122);
imshow(final_out); title('processed frame');
figname = [output_dir outname '_invout.png'];
exportgraphics(figs, figname);

% % save results
imwrite(uint8(final_out*255),[output_dir outname '_final.png']);

