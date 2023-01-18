function px_size = get_pixel_size(range, zoom)

    p00 =   -0.003657;
    p10 =   4.707e-07;
    p01 =   1.791e-05;
    p20 =  -1.893e-11;
    p11 =   -1.84e-09;
    p02 =  -1.378e-08;
    p21 =   7.507e-14;
    p12 =   6.222e-13;
    p03 =   5.017e-12;

    px_size = p00 + p10*zoom + p01*range + p20*zoom^2 + p11*zoom*range + p02*range^2 + p21*zoom^2*range + p12*zoom*range^2 + p03*range^3;

end