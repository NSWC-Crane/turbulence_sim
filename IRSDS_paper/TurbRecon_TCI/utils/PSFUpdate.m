function out = PSFUpdate(H,lambda,w,havr,tol,max_itr)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% update the dictionary for faster converge
% Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan
% Copyright 2020
% Purdue University, West Lafayette, In, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
change = inf;
itr = 0;
sizeH = size(H);
vsize = sizeH(1)*sizeH(2);
v = [];
havr = reshape(havr, [vsize,1]);

for i = 1:sizeH(3)
    v = [v; reshape(H(:,:,i), [sizeH(1)*sizeH(2) 1])];
end

while (change > tol && itr < max_itr)
    
    vold = v;
    
    v = 4*G(F(v,vsize,havr,lambda),w,vsize) - 2*G(v,w,vsize) - 2*F(v,vsize,havr,lambda) + v;

    change = sqrt(sum((v-vold).^2));
    
    itr = itr + 1;
    
end

for i=1:sizeH(3)
    out(:,:,i) = reshape(v((i-1)*sizeH(1)*sizeH(2)+1:i*sizeH(1)*sizeH(2)),[sizeH(1) sizeH(2)]);
end

end

function out = F(v,vsize,havr,lambda)

out=zeros(length(v),1);

for i = 1:vsize:length(v)
    out(i:i+vsize-1) = (lambda * v(i:i+vsize-1) + havr) / (lambda + 1);
    out(i:i+vsize-1) = out(i:i+vsize-1) ./ sum(out(i:i+vsize-1));
end

end

function out = G(v,w,vsize)

out = [];

avr = zeros(vsize,1);

for i = 1:length(w)
    avr = avr + w(i)*v((i-1)*vsize+1:i*vsize);
end

for i = 1:length(v)/vsize
    out = [out;avr];
end

end