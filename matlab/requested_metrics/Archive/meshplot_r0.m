% 3D Mesh for Fried parameter

[x,y] = meshgrid(linspace(.1e-14,10e-14,400), 600:5:1000);
lambda = .525e-6;
b0 = 0.158625;
k=2*pi/lambda;
r0 = (b0*k^2*x.*y).^-0.6;

figure()
mesh(x,y,r0, 'FaceAlpha', '0.5')
xlabel('Cn_2')
ylabel('Range (m)')
zlabel('Fried Parameter r_0')



