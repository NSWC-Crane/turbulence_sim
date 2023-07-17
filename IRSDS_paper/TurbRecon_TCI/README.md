# Image Reconstruction of Static and Dynamic Scenes through Anisoplanatic Turbulence

This repository contains the code for the following paper:

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan, ‘‘Image reconstruction of static and dynamic scenes through anisoplanatic turbulence’’, IEEE Trans. Computational Imaging, vol. 6, pp. 1415-1428, Oct. 2020.

How to use: 
  - Check the demo.m file
  - We've prepared a few sets of parameters for low, medium and high turbulence level (corresponding to D/r0 = 1.5, 3, and 4.5)
  
Packages included
  - Algorithm for image reconstruction through atmospheric turbulence, developed by Purdue i2Lab
  - Plug and Play ADMM, developed by Purdue i2Lab
  - Optical flow, developed by Ce Liu (Microsoft Research New England) https://people.csail.mit.edu/celiu/OpticalFlow/

The data used in the paper can be downloaded here: 

https://drive.google.com/file/d/1VsQyrPexjAXegAXAx7A5CmKB0F4hw_O7/view?usp=sharing

If you find our work helpful in your research, please consider cite our paper

```
@ARTICLE{mao_tci,
  author={Zhiyuan Mao and Nicholas Chimitt and Stanley H. Chan},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Image Reconstruction of Static and Dynamic Scenes Through Anisoplanatic Turbulence}, 
  year={2020},
  volume={6},
  pages={1415-1428},
  doi={10.1109/TCI.2020.3029401}
  }
```
  
Please also check out our other work on Atmospheric Turbulence: 
  - Project page
  
    https://engineering.purdue.edu/ChanGroup/project_turbulence.html
  
  - Simulator: 
  
    https://www.mathworks.com/matlabcentral/fileexchange/78742-atmospheric-turbulence-simulator-for-image-reconstruction
    
  
## LICENSE

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
