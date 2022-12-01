function [dirBase, dirSharp, basefileN, ImgNames] = GetImageInfo(rangeV, zoomV)

dirBase = "C:\Data\JSSAP\sharpest\z" + num2str(zoomV);

if rangeV < 1000
    dirSharp = dirBase + "\0" + num2str(rangeV);
    basefileN = "baseline_z" + num2str(zoomV) + "_r0" + num2str(rangeV) + ".png";
else
    dirSharp = dirBase + "\" + num2str(rangeV);
    basefileN = "baseline_z" + num2str(zoomV) + "_r" + num2str(rangeV) + ".png";
end

iFiles = dir(fullfile(dirSharp, '*.png'));
ImgNames = {iFiles(~[iFiles.isdir]).name};

end