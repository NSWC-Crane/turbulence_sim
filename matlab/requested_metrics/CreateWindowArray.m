function [winImgArray] = CreateWindowArray(img,windowsize,column, pixelScan, lastPixel)
for row = 1:pixelScan:lastPixel
    winImgArray(:,:,row) = img(row:row+windowsize-1,column:column+windowsize-1);
end

end