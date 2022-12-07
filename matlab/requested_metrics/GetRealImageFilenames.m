function [dirSharp, ImgNames] = GetRealImageFilenames(data_root, rangeV, zoomV)

dirBase = data_root + "sharpest\z" + num2str(zoomV);

if rangeV < 1000
    dirSharp = dirBase + "\0" + num2str(rangeV);
else
    dirSharp = dirBase + "\" + num2str(rangeV);
end

iFiles = dir(fullfile(dirSharp, '*.png'));
ImgNames = {iFiles(~[iFiles.isdir]).name};

end