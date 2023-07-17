function [z] = reference_frame_func( num_frames, frame_stack, beta, area, special_i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reference_frame_func implements the reference frame method described in
% section III.A of our paper. 
% 
% Input:    num_frames  - Number of frames to use for reference frame
%           frame_stack - Input sequence
%           beta        - Decaying weight for reference frame
%           area        - Search window size for reference frame
%           special_i   - Specify a frame to use (for dynamic scene)
% Output:   z           - Reference frame
%
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, IN, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    addpath(genpath('./utilities/'));

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

    %=== Define spatial window ====%
    K1 = 5;
    h  = fspecial('gaussian',[2*K1+1, 2*K1+1], sqrt((K1^2)/(log(100))));
    ind_z = 1;

    for fid = start_id:end_id
        %==== Load Data ====%
        tmp = frame_stack(:,:,fid);

        [rows,cols] = size(tmp);

        %==== Get neighboring frames ====%
        y = zeros(rows,cols,num_frames);
        v = zeros(rows,cols,num_frames);
        for k=0:num_frames-1
            tmp = frame_stack(:,:,fid-shift_fr+k);
            y(:,:,k+1) = tmp;
            v(:,:,k+1) = stdfilt(tmp,ones(2*K1+1)).^2;
        end

        y_len = size(y,3);
        y_mean = mean(y,3);

        %==== Compute Distance ====%
        w0   = zeros(rows,cols,num_frames);
        [X,Y] = meshgrid(-area:area,-area:area);
        pos  = numel(X);
        
        for k=1:num_frames
            fprintf('\b\b\b\b\b %3.0f%%', k/num_frames*100);
            dtmp = zeros(rows,cols,pos);
            for ii = 1:pos
                ip = X(ii);
                jp = Y(ii);
                if special_i >= 0
                    dtmp(:,:,ii) = imfilter(( y(:,:,special_i)-circshift( y(:,:,k),[ip,jp])).^2, h, 'symmetric')./(circshift(v(:,:,k),[ip,jp])+1e-6);
                else
                    dtmp(:,:,ii) = imfilter(( y_mean-circshift( y(:,:,k),[ip,jp])).^2, h, 'symmetric')./(circshift(v(:,:,k),[ip,jp])+1e-6);                    
                end
            end
            w0(:,:,k) = min(dtmp,[],3);
        end

        %==== Averaging ====%
        w = exp(-beta*w0);
        z_num = sum(y.*w,3);
        z_den = sum(w,3);
        z(:,:,ind_z) = z_num./z_den;
        ind_z = ind_z + 1;
    end
end

