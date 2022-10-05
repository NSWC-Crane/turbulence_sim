function [dirModBase, dirSharp, basefileN, ImgNames] = GetImageInfoMod(rangeV, zoomV)

dirBase = "C:\Data\JSSAP\sharpest\z" + num2str(zoomV);
dirModBase = "C:\Data\JSSAP\modifiedBaselines";

if rangeV < 1000
    dirSharp = dirBase + "\0" + num2str(rangeV);
    basefileN = "Mod_baseline_z" + num2str(zoomV) + "_r0" + num2str(rangeV) + ".png";
else
    dirSharp = dirBase + "\" + num2str(rangeV);
    basefileN = "Mod_baseline_z" + num2str(zoomV) + "_r" + num2str(rangeV) + ".png";
end

iFiles = dir(fullfile(dirSharp, '*.png'));
ImgNames = {iFiles(~[iFiles.isdir]).name};

end