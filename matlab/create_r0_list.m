format long g
format compact
clc
close all
clearvars

% get the location of the script file to save figures
full_path = mfilename('fullpath');
[startpath,  filename, ext] = fileparts(full_path);
plot_num = 1;

commandwindow;

%%
rangeV = 600:50:1000;

wvl = 525e-9;

k = 2*pi/wvl;
b0 = 0.158625;

cn2_1 = [5e-16,1e-15,4e-15,7e-15,1e-14,2e-14,3e-14,4e-14,5e-14,6e-14,7e-14,8e-14,9e-14,1e-13,4e-13,7e-13,1e-12];
cn2_2 = [2e-13,3e-13,5e-13,6e-13,8e-13,9e-13,2e-12,3e-12,4e-12,5e-12,6e-12,7e-12,8e-12,9e-12,1e-11];
cn2_3 = [2e-15,3e-15,5e-15,6e-15,8e-15,9e-15,6e-16,7e-16,8e-16,9e-16,4e-16];
cn2V = sort(cat(2,cn2_1, cn2_2, cn2_3));

for L = rangeV
    
    for Cn2 = cn2V
    
    
        r0 = exp(-0.6 * log(b0 * k * k * Cn2 * L));
        
        fprintf('%d, %3.2e, %9.8f\n', L, Cn2, r0);
        
    end
    
end

