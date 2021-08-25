# TurbulenceSim_v1Public
This repository contains the code for the following paper:

Nicholas Chimitt and Stanley H. Chan "Simulating anisoplanatic turbulence by sampling intermodal and spatially correlated Zernike coefficients," Optical Engineering 59(8), Aug. 2020

How to use:

- There are two example files provided, the "Example_1_full.py" file, and the "Example_2_tiltonly.py" files. The tilt-only file is simpler, yet the Example_1_full offers the full version of the simulator described in our work. The Example_1_full is a documented typical usage case.
    
- The "TurbSim_v1_main.py" file forms the main body of the functionality, calling upon "Motion_Compensate.py" and "Integrals_Spatial_Corr.py" as needed. The functions therein are documented to help in their interpretability.

Packages included

- We use a function to simplify some indexing "nollToZernInd" (Tim van Werkhoven, Jason Saredy; https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py)

We would also encourage those interested in this simulator to use the Python version, which offers more flexibility and ease of use than its MATLAB counterpart.

If you find our work helpful in your research, please consider citing our paper

```
@article{chimitt_chan_sim,
author = {Nicholas Chimitt and Stanley H. Chan},
title = {{Simulating anisoplanatic turbulence by sampling intermodal and spatially correlated Zernike coefficients}},
volume = {59},
journal = {Optical Engineering},
number = {8},
publisher = {SPIE},
pages = {1 -- 26},
keywords = {atmospheric turbulence, simulator, anisoplanatism, Zernike polynomials, spatially varying blur, Turbulence, Point spread functions, Optical engineering, Wave propagation, Optical transfer functions, Atmospheric propagation, Computer simulations, MATLAB, Visualization},
year = {2020},
doi = {10.1117/1.OE.59.8.083101},
URL = {https://doi.org/10.1117/1.OE.59.8.083101}
}
```


