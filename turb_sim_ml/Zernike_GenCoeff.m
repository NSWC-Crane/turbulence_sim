function C = Zernike_GenCoeff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Zernike coefficients
%
% Nick Chimitt and Stanley Chan
% version: 7-Aug-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mytable = [
0 0 1;
1 1 2;
1 1 3;
2 0 4;
2 2 5;
2 2 6;
3 1 7;
3 1 8;
3 3 9;
3 3 10;
4 0 11;
4 2 12;
4 2 13;
4 4 14;
4 4 15;
5 1 16;
5 1 17;
5 3 18;
5 3 19;
5 3 20;
5 5 21;
6 0 22;
6 2 23;
6 2 24;
6 4 25;
6 4 26;
6 6 27;
6 6 28;
7 1 29;
7 1 30;
7 3 31;
7 3 32;
7 5 33;
7 5 34;
7 7 35;
7 7 36];

n = mytable(:,1);
m = mytable(:,2);

C = zeros(36,36);
for i=1:36
    for j=1:36
        ni = n(i); nj = n(j);
        mi = m(i); mj = m(j);
        if (mod(i-j,2)~=0)||(mi~=mj)
            C(i,j) = 0;
        else
            den = gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2)*gamma((ni+nj+23/3)/2);
            C(i,j) = 0.0072*(-1)^((ni+nj-2*mi)/2)*sqrt((ni+1)*(nj+1))*pi^(8/3)*...
                gamma(14/3)*gamma((ni+nj-5/3)/2)/den;
        end
    end
end
C(1,1) = 1;
