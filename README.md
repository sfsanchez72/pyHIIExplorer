# pyHIIExplorerV3

pyHIIexplorer V3 is a package for detecting and extracting physical properties from HII regions from integral field spectroscopy (IFS) datacubes or/and images. This version is based on pyHIIexplorer written by [Espinosa-Ponce, C. et al 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.1622E/abstract). and HIIexplorer written by [Sánchez, S. F. et al 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...546A...2S/abstract).


## Quick Description

We present a new tool called pyHIIexplorer V2 to explore and exploit the information provided by integral-field spectroscopy technique in the optical range. The code detects clumpy regions of Halpha maps (candidates for HII regions) and extracts as much spectroscopic information as possible (for both the underlying stellar populations and emission lines). Simultaneously during the detection and extraction of the clumpy regions, pyHIIexplorer V3 builds a diffuse ionized gas model (DIG). The construction of DIG will allow us to decontaminate the information of the HII regions candidates. 

## Designed for all cases

The code was intented for high resolution data like MUSE, but but in reality you can adjust the parameters to detect and extract information from surveys such as SAMI, ManGA and CALIFA.
The code is versatile and allows the user to: perform the detection and extraction in a only step in two steps from a datacube, Pipe3D or an image.

## Input parameters

1. Name (galaxy name or target)
2. Input file (File of the galaxy, can be a cube, a cube pipe3D, or an image. If the input file is a cube )
3. 



