% SSIM of spatial frequency with complex numbers
% Read in images, take laplacian (optional), call SSIM function.

% The mean of the imaginary part of the fft of the images is in range
% of e-19.
% Adjusted dynamicRange so that C1 and C2 won't result in a metric 
% that approaches 1 in all cases.

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
dirOut = data_root + "modifiedBaselines\SimImgs_VaryingCn2\ssim_plots\";
dynamicRange = 0.75;

for index = 1:length(realFiles)

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
    
    ImageR = ImageR(2:end-1, 2:end-1);
    
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
        
        ImageSim = ImageSim(2:end-1, 2:end-1);

        % Pull out Cn2 value from filename
        cn2 = split(simNamelist{k}, 'c');
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
        
        turb_metric = turbulence_metric_noBL(ImageR, ImageSim);
    
        Tc(k,:) = {fileS, cn2, ssimFC, ssimMagn, ssimPhase, ssimReal, ssimImg, turb_metric};
        
    end
    
    varnames = {'filename', 'Cn2', 'ssimFC', 'ssimMagn', 'ssimPhase','ssimReal', 'ssimImg', 'turb_metric'}; 
    Tc = renamevars(Tc, Tc.Properties.VariableNames, varnames);
    Tc.filename = string(Tc.filename);
    Tc = sortrows(Tc, "Cn2");
       
    % Semilogx plot
    upY = 0.4;
    ffh = figure();
    hold on
    s1 = semilogx(Tc.Cn2, abs(Tc.ssimFC),'-og',...  % Fully complex
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.ssimFC);
    s1a = stem(Tc.Cn2(indI), 1, 'g', 'filled');
    str = "ssimFC: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    hold on
    grid on
    s2 = semilogx(Tc.Cn2, Tc.ssimMagn,'-ob',...    % Magnitude
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.ssimMagn);
    s2a = stem(Tc.Cn2(indI), 1, 'b', 'filled');
    str = "ssimMagn: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    hold on
    s3 = semilogx(Tc.Cn2, Tc.ssimPhase,'-oc',...  % Phase
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.ssimPhase);
    s3a = stem(Tc.Cn2(indI), 1, 'c', 'filled');
    str = "ssimPhase: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    hold on
    s4 = semilogx(Tc.Cn2, Tc.ssimReal,'-or',...    % Real component
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.ssimReal);
    s4a = stem(Tc.Cn2(indI), 1, 'r', 'filled');
    str = "ssimReal: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    hold on
    s5 = semilogx(Tc.Cn2, Tc.ssimImg,'-om',...  % Imaginary component
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.ssimImg);
    s5a = stem(Tc.Cn2(indI), 1, 'm', 'filled');
    str = "ssimImg: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    hold on
    s6 = semilogx(Tc.Cn2, Tc.turb_metric,'-o', 'Color', [0.4940 0.1840 0.5560],...  % Imaginary component
                'LineWidth',2,...
                'MarkerSize',3);
    [~, indI] = max(Tc.turb_metric);
    s6a = stem(Tc.Cn2(indI), 1, 'filled', 'Color', [0.4940 0.1840 0.5560]);
    str = "turb_metric: Max metric at Cn2 " + num2str(Tc.Cn2(indI));
    annotation('textbox',[.74 .5 .3 upY], ...
        'String',str,'EdgeColor','none')
    upY = upY-0.05;
    
    
    
    stem(realcn2(index), 1, 'k', 'filled');
    
    set(gca,'xscal','log')

    xlabel("Cn2 (Real is " + num2str(realcn2(index)) + ")")
    xlim([min(Tc.Cn2), max(Tc.Cn2)])
    ylabel('SSIM Index')
    legend([s1,s2,s3,s4,s5,s6], {'SSIM Fully Complex', 'SSIM Magnitude Only', 'SSIM Phase Only', 'SSIM Separate Real', 'SSIM Separate Img', 'turbMetric'}, 'location', 'southeastoutside')
    %legend('SSIM Fully Complex', 'SSIM Magnitude Only', 'SSIM Separate Real', 'location', 'southeastoutside')
    title("Range " + rng + " Zoom " + zm  + " with Measured Cn2 of " + num2str(realcn2(index)))
    width=900;
    height=500;
    fileN = dirOut + num2str(index) + "_plotLog.png";
    set(gcf,'position',[10,10,width,height])
%     f = gcf;
%     exportgraphics(f,fileN,'Resolution',300)
    %savefig(ffh,fileNf)
    % close(ffh)
    hold off
end
        
