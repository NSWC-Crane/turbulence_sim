% Save all files in one sharpest folder as green channel only images
% (look like simulated)

% Reals vary
clear
clc

rdir = "C:\Data\JSSAP\sharpest\z3000\0650\";
dirOut = "C:\Data\JSSAP\modifiedBaselines\NewSimulations\AllReals\R650Z3000\";
xstr = ["i00","i01","i02","i03","i04","i05","i06","i07","i08","i09","i10",...
         "i11","i12","i13","i14","i15","i16","i17","i18","i19"];

% get metrics for all reals in rdir as compared to simulated image sfile.
image_ext = '*.png';
rlisting = dir(strcat(rdir, '/', image_ext));

for idx = 1:length(rlisting)
    % Read in images and clip border
    rname = fullfile(rdir, '/', rlisting(idx).name);
    RealImg = double(imread(rname));
    [img_h,img_w,img_d] = size(RealImg);
    if img_d > 1
        RealImg = RealImg(:,:,2);
    end
    pathI = dirOut + xstr(idx) + ".png";
    RealImg = uint8(RealImg);
    imwrite(RealImg, pathI);
     
end


