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

clear all
close all
clc

addpath(genpath('./downloads'))
addpath(genpath('./utils'))

% set input path and load data
input_dir = './data/scene4/';
imgs = dir([input_dir '*.png']);
for i = 1:length(imgs)
    temp = imread([input_dir imgs(i).name]);
    if size(temp,3) == 3
        temp = rgb2gray(temp);
    end
    stack(:,:,i) = double(temp);
end

% load parameters
load ./utils/medium.mat

% registration step
fprintf('===========================\n');
fprintf('TurbRecon_v1, Purdue University, 2020\n');
fprintf('===========================\n');

fprintf('Part 1: Image registration\n')
[luck_out, reg_out, reg_stack] = registration_main(stack, reg_parameters);
fprintf('\n');


% blind deblurring step
fprintf('Part 2: Blind deconvolution \n')
[final_out, hnew] = deblur_main(luck_out,deblur_parameters);

fprintf('\n');
fprintf('Done. Display results.\n');

figure;
subplot(121);
imshow(stack(:,:,end/2), []); title('input: frame 50/100');
subplot(122);
imshow(final_out); title('processed: frame 50/100');

% % save results
% imwrite(uint8(final_out*255),['./data/scene4_out.png']);

