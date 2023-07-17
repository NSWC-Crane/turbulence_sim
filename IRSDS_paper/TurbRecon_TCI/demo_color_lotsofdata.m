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

start_time = clock;

addpath(genpath('./downloads'))
addpath(genpath('./utils'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check to see if image is color with 1 or 0 switch, 1 is yes
color_img = 1;
% load parameters
load ./utils/medium.mat
% set how many files you want to read in -1 is all
reg_parameters.id_2 = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set input path and load data by getting user to give directory
myDir = uigetdir; %gets directory
% set output directory
output_dir = [myDir '/output_5'];
fprintf(1, 'Now reading %s\n', myDir);
% finds all directories with fp in directory name
myFiles = dir(fullfile(myDir, 'fp*'));
if ~exist(output_dir, 'dir')
   mkdir(output_dir)
end

% for each directory in myFiles
for k = 1:length(myFiles)
    dir_name = myFiles(k).name;
    input_dir = [myDir '/' dir_name '/'];
    fprintf('===========================\n');
    fprintf('===========================\n');
    fprintf(1, 'Now reading %s\n', dir_name);
    fprintf('===========================\n');
    fprintf('===========================\n');
    outname = dir_name;

    %input_dir = './data/scene4/';
    imgs = dir([input_dir '*.png']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % check how many colors there are though it should always be 3
    temp_color = imread([input_dir imgs(1).name]);

    if color_img == 1
        for j = 1:size(temp_color,3)
            [x , y, z] = size(temp_color);
            stack = zeros(x, y, length(imgs));
            for i = 1:length(imgs)
                temp = imread([input_dir imgs(i).name]);
                if size(temp,3) == 3
                    %display(temp)
                    temp = temp(:,:, j);
                end
                stack(:,:,i) = double(temp);
            end

            if j == 1
                color_name = 'red';
            elseif j == 2
                color_name = 'green';
            elseif j == 3
                color_name = 'blue';
            end
            disp(color_name);

            % registration step
            fprintf('===========================\n');
            fprintf('TurbRecon_v1, Purdue University, 2020\n');
            fprintf('===========================\n');

            fprintf('Part 1: Image registration\n')
            [luck_out, reg_out, reg_stack] = registration_main(stack, reg_parameters);
            fprintf('\n');

            if j == 1
                luck_red = luck_out;
                reg_red = reg_out;
            elseif j == 2
                luck_green = luck_out;
                reg_green = reg_out;
            elseif j == 3
                luck_blue = luck_out;
                reg_blue = reg_out;
            end

            % blind deblurring step
            fprintf('Part 2: Blind deconvolution \n')
            [final_out, hnew] = deblur_main(luck_out,deblur_parameters);

            fprintf('\n');
            fprintf('Done. Save results for later.\n');

            if j == 1
                final_red = final_out;
            elseif j == 2
                final_green = final_out;
            elseif j == 3
                final_blue = final_out;
            end
        fprintf('\n');
        end

        %make final luck, reg, and final img
        luck_out_3D(:, :, 1) = luck_red;
        luck_out_3D(:, :, 2) = luck_green;
        luck_out_3D(:, :, 3) = luck_blue;
        reg_out_3D(:, :, 1) = reg_red;
        reg_out_3D(:, :, 2) = reg_green;
        reg_out_3D(:, :, 3) = reg_blue;
        final_out_3D(:, :, 1) = final_red;
        final_out_3D(:, :, 2) = final_green;
        final_out_3D(:, :, 3) = final_blue;
        
        % save results
        imwrite(uint8(luck_out_3D*255),[output_dir '/' outname '_luck_color.png']);
        imwrite(uint8(reg_out_3D*255),[output_dir '/' outname '_reg_color.png']);
        imwrite(uint8(final_out_3D*255),[output_dir '/' outname '_final_color.png']);

    else
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Will need to edit this to deal with new format

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
        figname = [output_dir '/' outname '_luckvreg_gray.png'];
        exportgraphics(figs2, figname);
        imwrite(uint8(luck_out*255),[output_dir '/' outname '_luck_gray.png']);
        imwrite(uint8(reg_out*255),[output_dir '/' outname '_reg_gray.png']);

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
        figname = [output_dir '/' outname '_invout_gray.png'];
        exportgraphics(figs, figname);

        % % save results
        imwrite(uint8(final_out*255),[output_dir '/' outname '_final_gray.png']);

    end
end

end_time = clock;
