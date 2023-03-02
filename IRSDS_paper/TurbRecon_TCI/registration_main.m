function [luck_out, reg_out, reg_stack] = registration_main(stack, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [luck_out, reg_out, reg_stack] = registration_main(stack, para)
% This function is the main routine for the image registration part
% (section III.A and III.B) in the paper

% Input: stack - Input sequence (typically 100 frames)
%        para  - Registration parameters (provided in the parameters folder)
%              - para.beta_ref  decaying weight for reference frame
%              - para.area_ref  search window size for reference frame
%              - para.beta1     weight for sharpness metric
%              - para.beta2     weight for geometric metric
%              - para.flow      parameters for optical flow
%              - para.id_1      used to select a set of frames to use
%              - para.id_2      used to select a set of frames to use
%
% Output: luck_out - Lucky region fusion result
%         reg_out - Estimated reference frame
%         reg_stack - Each registered frame
%
% Reference: We use the optical flow function from: 
%            Ce Liu, "Beyond pixels: Exploring new representations and
%            application for motion analysis", Ph.D. dissertation, MIT, 2009
% 
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

beta_ref = para.beta_ref;
area_ref = para.area_ref;
beta1 = para.beta1;
beta2 = para.beta2;
flow_para = para.flow;

id_1 = para.id_1;
id_2 = para.id_2;
if id_2 == -1
    id_1 = 1;
    id_2 = size(stack,3);
end

stack = stack(:,:,id_1:id_2);

if para.special_i == 0
    t_grad_prev = -1;
    for i = 1:size(stack,3)
        thing = stack(:,:,i);
        t_grad = sum(sum(abs(imgradient(thing))));
        if t_grad > t_grad_prev
            t_grad_prev = t_grad;
            special_i = i;
        end
    end
else
    special_i = para.special_i;
end
fprintf('Extracting reference frame %3.0f%%\n', 0);
z = reference_frame_func( length(id_1:id_2), stack, beta_ref, area_ref, special_i);
fprintf('\n');


fprintf('Clean up reference frame     \n');
a = padarray(z,[20 20],'replicate');
padstack = padarray(stack,[20 20],'replicate');

z = a/255;

if para.deblur == 1
    deblur_para = para.deblur_para;
    method = 'BM3D';
    lambda = 0.00005;

    %optional parameters
    opts.rho     = 0.05;
    opts.gamma   = 1;
    opts.max_itr = 30;
    opts.print   = false;
    h = fspecial('gaussian',[21 21],deblur_para);

    z = PlugPlayADMM_deblur(z,h,lambda,method,opts);
end
fprintf('\n');

fprintf('Computing optical flow     %3.0f%%\n', 0);
[reg_stack] = wrapper_optflow( padstack/255, z, flow_para);
fprintf('\b\b\b\b\b 100%% \n');



fprintf('Lucky region fusion        %3.0f%%\n', 0);
[ luck_stack ] = lucky_region_func( length(id_1:id_2), reg_stack,z,...
    beta1, beta2);
fprintf('\b\b\b\b\b 100%% \n');



luck_out = luck_stack(21:size(stack,1)+20,21:size(stack,2)+20,:);
reg_stack = reg_stack(21:size(stack,1)+20,21:size(stack,2)+20,:);
reg_out = z(21:size(stack,1)+20,21:size(stack,2)+20);
end
