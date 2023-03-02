function [ out,num,den ] = lucky_region_func( num_frames, frame_stack, ref, beta1, beta2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lucky_region_func implements the lucky region fusion method described in
% section III.B of our paper. 
% 
% Input:    num_frames  - Number of frames to use for reference frame
%           frame_stack - Input sequence
%           ref         - Reference frame
%           beta1       - Weight for sharpness metric
%           beta2       - Weight for geometric metric
% Output:   out         - lucky frame stack
%
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    total_frames = size(frame_stack,3);
    if mod(num_frames,2)
        start_id = (num_frames + 1)/2;
        end_id = total_frames - (num_frames - 1)/2;
    else
        start_id = num_frames/2 +1;
        end_id = total_frames - (num_frames/2 - 1);
    end

    if mod(num_frames,2)
        shift_fr = (num_frames - 1)/2;
    else
        shift_fr = num_frames/2;
    end
    
    out = zeros([size(frame_stack,1), size(frame_stack,2), length(start_id:end_id)]);
    out_ind = 1;

    for ii = start_id:end_id
        fprintf('\b\b\b\b\b %3.0f%%', (ii-start_id+1)/num_frames*100);
        tmp = frame_stack(:,:,ii);

        [rows,cols] = size(tmp);
        y     = zeros(rows,cols,num_frames);
        Ygrad = zeros(rows,cols,num_frames);
        
        for k=0:num_frames-1
            tmp = frame_stack(:,:,ii-shift_fr+k);
            y(:,:,k+1) = tmp;
            [Grad,~] = imgradient(tmp);
            Ygrad(:,:,k+1) = Grad;
        end
        
        y_len = size(y,3);
        yref = ref;

        h    = ones(15,15); h = h/sum(h(:));
        score_match = zeros(rows,cols,num_frames);
        score_sharp = zeros(rows,cols,num_frames);
        for k=1:num_frames            
            score_match(:,:,k) = imfilter( ( yref - y(:,:,k) ).^2, h, 'symmetric');
            score_sharp(:,:,k) = imfilter( abs(Ygrad(:,:,k)), h, 'symmetric' );
        end

        score_mix = (score_sharp.^beta1).*exp(-beta2*score_match)+eps;

        num = sum(score_mix.*y,3);
        den = sum(score_mix,3);
        z   = num./den;

        out(:,:,out_ind) = z;
        out_ind = out_ind + 1;
    end


end

