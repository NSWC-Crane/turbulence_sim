function [ out_stack ] = wrapper_optflow( raw_stack, ref_stack, para )
% wrapper of the optical flow function    
    len_raw = size(raw_stack,3);
    len_ref = size(ref_stack,3);
    if len_raw == len_ref
        mult_out = 1;
        outstack = zeros(size(ref_stack));
    else
        mult_out = 0;
        outstack = zeros(size(raw_stack));
    end    

    for fid = 1:len_raw
        fprintf('\b\b\b\b\b %3.0f%%', fid/len_raw*100);
        
        
        if mult_out
            [vx,vy,warpI2] = Coarse2FineTwoFrames(ref_stack(:,:,fid),...
                raw_stack(:,:,fid),para);
        else
            [vx,vy,warpI2] = Coarse2FineTwoFrames(ref_stack,...
                raw_stack(:,:,fid),para);
        end
        
        out_stack(:,:,fid) = warpI2;
    end
    


end

