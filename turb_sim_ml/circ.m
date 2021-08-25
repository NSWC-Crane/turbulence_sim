function W = circ(rowsW,colsW)
if rowsW~=colsW
    error('rows not equal to cols.');
end

if (mod(rowsW,2)==0)&&(mod(colsW,2)==0)
    [x,y]     = meshgrid(-rowsW/2+1/2:rowsW/2-1/2,-colsW/2+1/2:colsW/2-1/2);
    [~,r]     = cart2pol(x,y);
    W         = zeros(rowsW,colsW);
    W(r<rowsW/2-1) = 1;
else
    error('rows and cols must be even.');
end