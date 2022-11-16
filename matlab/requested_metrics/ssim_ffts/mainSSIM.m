% SSIM of spatial frequency with complex numbers
% Read in images, take laplacian (optional), call SSIM function.

% The mean value of the imaginary part of the fft of the images is in range
% of e-19.
% Adjusted dynamicRange so that C1 and C2 don't cause the metric to approach
% 1 in all cases.

clearvars
clc

% Setup directories
platform = string(getenv("PLATFORM"));
if(platform == "Laptop")
    data_root = "D:\data\turbulence\";
elseif (platform == "LaptopN")
    data_root = "C:\Projects\data\turbulence\";
else   
    data_root = "C:\Data\JSSAP\";
end

% Setup real files to choose
realFiles = [data_root + "sharpest\z3000\0700\image_z02997_f47045_e05982_i00.png"; % Cn2: 5E-15
             data_root + "sharpest\z4000\0650\image_z03995_f47715_e09088_i05.png"; % Cn2: 1.89E-15
             data_root + "sharpest\z2500\0750\image_z02498_f46610_e06279_i02.png"; % Cn2: 9.1E-15
             data_root + "sharpest\z3500\0700\image_z03497_f47470_e06081_i03.png"; % Cn2: 5.44E-15
             data_root + "sharpest\z2000\0750\image_z01999_f46254_e05585_i09.png"  % Cn2: 1.09E-14
             ];
realcn2 = [5e-15; 1.89e-15; 9.1e-15; 5.44e-15; 1.09e-14];
dirSims = data_root + "modifiedBaselines\SimImgs_VaryingCn2\";
dynamicRange = 0.75;

% User selects real file to use by defining index
% realFiles(2) is the second file on the list.
index = 2;

% Generate the simimulated image set based on zoom and range of the selected 
% real file (only use one of each Cn2 - use N0.png)
simFiles = dir(fullfile(dirSims, '*.png'));
SimImgNames = {simFiles(~[simFiles.isdir]).name};
simNamelist = []; % list of all simulated image files at this zoom/range
ind = 1;
% Filter by range and zoom to get file names of range/zoom
zm = split(realFiles(index), "z");
zm = zm(2);
zm = split(zm, "\");
zm = zm(1);

rng = split(realFiles(index), "\im");
rng = split(rng(1),"\");
rng = char(rng(end));
if rng(1) == '0'
    rng = string(rng(2:4));
else
    rng = string(rng(1));
end

patt1 = "r" + rng + "_z" + zm;
patt2 = "_N0.png";  % Only selecting one simulated image per Cn2 value
for i = 1:length(SimImgNames)
    if contains(SimImgNames{:,i},patt1)
        if contains(SimImgNames{:,i},patt2)
            simNamelist{ind,1} = SimImgNames{:,i};
            %display(namelist{ind})
            ind = ind +1;
        end
    end
end

% OPTIONS in processing images
getLapl = true;

% Laplacian kernel
lKernel = 0.25*[0,-1,0;-1,4,-1;0,-1,0];

% Read in real image - green channel only
ImageR = double(imread(realFiles(index))); 
ImageR= ImageR(:,:,2); 
if getLapl == true
    % Find Laplacian of Real Image
    ImageR = conv2(ImageR, lKernel, 'same'); 
end

% Loop through simulated image list simNamelist
% Collect filename, Cn2, metrics
Tc = table;

for  k = 1:length(simNamelist)
    fileS = fullfile(dirSims, simNamelist{k});
    ImageSim = double(imread(fileS)); 

    if getLapl == true
        % Find Laplacian of Real Image
        ImageSim = conv2(ImageSim, lKernel, 'same'); 
    end
    
    % Pull out Cn2 value from filename
    cn2 = split(fileS, 'c');
    cn2 = char(cn2(2));
    cn2 = string(cn2(1:4));
    cn2N = insertAfter(cn2, "e","-");
    cn2 = str2double(cn2N);

    % CALL FUNCTIONS - input arguments ImageR, ImageSim
    % Functions:
    % 1. Fully complex (SSIM_FFT_fullComplex)
    % 2. Separate real and imaginary results (2 SSIMs) (SSIM_FFT_sepComplex)
    % 3. Use the magnitude of the FFT and the SSIM formula (SSIM_FFT_magn)
    ssimFC = SSIM_FFT_fullComplex(ImageR, ImageSim, dynamicRange);
    [ssimReal, ssimImg] = SSIM_FFT_SepRealImg(ImageR, ImageSim, dynamicRange);
    [ssimMag, ssimPhase] = SSIM_FFT_SepMagPhase(ImageR, ImageSim, dynamicRange);
    ssimMagn = SSIM_FFT_magn(ImageR, ImageSim, dynamicRange);

    Tc(k,:) = {fileS, cn2, ssimFC, ssimMagn, ssimPhase, ssimReal, ssimImg};
    
end

varnames = {'filename', 'Cn2', 'ssimFC', 'ssimMagn', 'ssimPhase','ssimReal', 'ssimImg'}; 
Tc = renamevars(Tc, Tc.Properties.VariableNames, varnames);
Tc.filename = string(Tc.filename);
Tc = sortrows(Tc, "Cn2");

% % Plot metrics as a function of Cn2
% ffg = figure();
% plot(Tc.Cn2, abs(Tc.ssimFC),'-og',...
%             'LineWidth',2,...
%             'MarkerSize',3)
% hold on
% grid on
% plot(Tc.Cn2, Tc.ssimMagn,'-ob',...
%             'LineWidth',2,...
%             'MarkerSize',3)
% hold on
% plot(Tc.Cn2, Tc.ssimPhase,'-oc',...
%             'LineWidth',2,...
%             'MarkerSize',3)
% hold on
% plot(Tc.Cn2, Tc.ssimReal,'-or',...
%             'LineWidth',2,...
%             'MarkerSize',3)
% hold on
% plot(Tc.Cn2, Tc.ssimImg,'-om',...
%             'LineWidth',2,...
%             'MarkerSize',3)
% xlabel("Cn2 (Real is " + num2str(realcn2(index)) + ")")
% xlim([min(Tc.Cn2), max(Tc.Cn2)])
% ylabel('SSIM Index')
% legend('SSIM Fully Complex', 'SSIM Magnitude Only', 'SSIM Phase Only','SSIM Separate Real', 'SSIM Separate Img','location', 'northeastoutside')
% width=900;
% height=500;
% % fileN = "C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2\ssim_plots\plot3.png";
% % fileNf = "C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2\ssim_plots\plot3.fig";
% set(gcf,'position',[10,10,width,height])
% % f = gcf;
% % exportgraphics(f,fileN,'Resolution',300)
% % 
% % savefig(ffg,fileNf)
% % close(ffg)
% hold off

% Semilogx plot
ffh = figure();
semilogx(Tc.Cn2, abs(Tc.ssimFC),'-og',...  % Fully complex
            'LineWidth',2,...
            'MarkerSize',3)
hold on
grid on
semilogx(Tc.Cn2, Tc.ssimMagn,'-ob',...    % Magnitude
            'LineWidth',2,...
            'MarkerSize',3)
% hold on
% semilogx(Tc.Cn2, Tc.ssimPhase,'-oc',...  % Phase
%             'LineWidth',2,...
%             'MarkerSize',3)
hold on
semilogx(Tc.Cn2, Tc.ssimReal,'-or',...    % Real component
            'LineWidth',2,...
            'MarkerSize',3)
% hold on
% semilogx(Tc.Cn2, Tc.ssimImg,'-om',...  % Imaginary component
%             'LineWidth',2,...
%             'MarkerSize',3)
xlabel("Cn2 (Real is " + num2str(realcn2(index)) + ")")
xlim([min(Tc.Cn2), max(Tc.Cn2)])
ylabel('SSIM Index')
%legend('SSIM Fully Complex', 'SSIM Magnitude Only', 'SSIM Phase Only','SSIM Separate Real', 'SSIM Separate Img','location', 'northeastoutside')
legend('SSIM Fully Complex', 'SSIM Magnitude Only', 'SSIM Separate Real', 'location', 'northeastoutside')
title("Range " + rng + " Zoom " + zm  + " with Measured Cn2 of " + num2str(realcn2(index)))
width=900;
height=500;
% fileN = "C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2\ssim_plots\plotLog3.png";
% fileNf = "C:\Data\JSSAP\modifiedBaselines\SimImgs_VaryingCn2\ssim_plots\plotLog3.fig";
set(gcf,'position',[10,10,width,height])
% f = gcf;
% exportgraphics(f,fileN,'Resolution',300)
% savefig(ffh,fileNf)
% close(ffh)
hold off
