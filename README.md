# pyHIIExplorerV3

pyHIIexplorer V3 is a package for detecting and extracting physical properties from HII regions from integral field spectroscopy (IFS) datacubes or/and images. This version is based on pyHIIexplorer written by [Espinosa-Ponce, C. et al 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.1622E/abstract). and HIIexplorer written by [Sánchez, S. F. et al 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...546A...2S/abstract).


## Quick Description

We present a new tool called pyHIIexplorer V2 to explore and exploit the information provided by integral-field spectroscopy technique in the optical range. The code detects clumpy regions of Halpha maps (candidates for HII regions) and extracts as much spectroscopic information as possible (for both the underlying stellar populations and emission lines). Simultaneously during the detection and extraction of the clumpy regions, pyHIIexplorer V3 builds a diffuse ionized gas model (DIG). The construction of DIG will allow us to decontaminate the information of the HII regions candidates. 

## Designed for all cases

The code was intented for high resolution data like MUSE, but but in reality you can adjust the parameters to detect and extract information from surveys such as SAMI, ManGA and CALIFA.
The code is versatile and allows the user to: perform the detection and extraction in a only step in two steps from a datacube, Pipe3D or an image.

## Input parameters

### Case 1: If the detection and extraction is in two steps

  #### Detection from a datacube flux_elines (bin/pyHIIdet_cube.py):

  1. name (Galaxy name or target)
  2. input file (Cube flux_elines of the galaxy)
  3. n_hdu (HDU data index)
  4. n_Ha (Index Halpha)
  5. n_eHa (Index Halpha error)
  6. FWHM (FWHM of the image)
  7. spax_sca (Spaxel scale)
  8. MUSE_1sig (1sig, if the value is -1 then the program use Index Halpha error)
  9. MUSE_1sig_V (1sig_continuum,  if the value is -1 then the program use Index Halpha error)
  10. plot (Save plot,  where 0=not 1=yes)
  11. refined (Refined detection)
  12. maps_seg (Save segmentation maps, 0=not 1=yes)
  13. DIG_type_weight (Create new DIG with weight system,  0=not 1=yes)
  14. max_size (Max_size, HIIregions)
  15. DIR (Where save outputfiles)
  
  #### Detection from a image (bin/pyHIIdet_img.py):
  
  1. name (Galaxy name or target)
  2. input file (Cube flux_elines of the galaxy)
  3. n_hdu (HDU data index)
  4. FWHM (FWHM of the image)
  5. spax_sca (Spaxel scale)
  6. MUSE_1sig (1sig, if the value is -1 then the program use Index Halpha error)
  7. MUSE_1sig_V (1sig_continuum,  if the value is -1 then the program use Index Halpha error)
  8. plot (Save plot,  where 0=not 1=yes)
  9. refined (Refined detection)
  10. maps_seg (Save segmentation maps, 0=not 1=yes)
  11. DIG_type_weight (Create new DIG with weight system,  0=not 1=yes)
  14. max_size (Max_size, HIIregions)
  15. DIR (Where save outputfiles)
  
  #### Extraction (bin/pyHIIext.py): 

  1. name (Galaxy name or target)
  2. input_file_fe (Input cube flux elines)
  3. input_file_ssp (Input cube ssp)
  4. input_file_sfh (Input file sfh)
  5. input_file_index (Input file indices)
  6. blobs_final (Blobs of detection HII regions (table))
  7. diff_points (Points diffuse (table))
  8. FWHM (FWHM of the image)
  9. spax_sca (Spaxel scale)
  10. plot (Save plot,  where 0=not 1=yes)
  11. def_DIG (Create new DIG with weight system only for fluxes,  0=not 1=yes)
  12. DIR (Where save outputfiles)
  
  #### Extraction (bin/pyHIIext_pipe3d.py): 

  1. name (Galaxy name or target)
  2. input_file (Input cube Pipe3d)
  3. n_hdu_fe (HDU flux_elines index)
  4. n_hdu_ssp (HDU single stellar populations index)
  5. n_hdu_sfh (HDU star formation history index)
  6. n_hdu_index (HDU Lick-indices index)
  7. blobs_final (Blobs of detection HII regions (table))
  8. diff_points (Points diffuse (table))
  9. FWHM (FWHM of the image)
  10. spax_sca (Spaxel scale)
  11. plot (Save plot,  where 0=not 1=yes)
  12. def_DIG (Create new DIG with weight system only for fluxes,  0=not 1=yes)
  13. DIR (Where save outputfiles)

### Case 2: If the detection and extraction is in one step from a datacube Pipe3D

  #### Detection and extraction (bin/pyHIIdet_ext_pipe3d.py):
  
  1. name (Galaxy name or target)
  2. input file (Cube flux_elines of the galaxy)
  3. n_hdu_fe (HDU flux_elines index)
  4. n_hdu_ssp (HDU single stellar populations index)
  5. n_hdu_sfh (HDU star formation history index)
  6. n_hdu_index (HDU Lick-indices index)
  7. n_Ha (Index Halpha)
  8. n_eHa (Index Halpha error)
  9. FWHM (FWHM of the image)
  10. spax_sca (Spaxel scale)
  11. MUSE_1sig (1sig, if the value is -1 then the program use Index Halpha error)
  12. MUSE_1sig_V (1sig_continuum,  if the value is -1 then the program use Index Halpha error)
  13. plot (Save plot,  where 0=not 1=yes)
  14. refined (Refined detection)
  15. maps_seg (Save segmentation maps, 0=not 1=yes)
  16. def_DIG (Create new DIG with weight system only for fluxes,  0=not 1=yes)
  17. max_size (Max_size, HIIregions)
  18. DIR (Where save outputfiles)

## Requerimemnts

- python 3.x
- numpy package
- astropy package
- scipy package
- matplotlib package
- skimage package
- argparse module 

## How to use it?

In the example file: bin/pyHIIdet_ext_pipe3d.py coudl find a implementation of full code with the file on data: NGC2906.Pipe3D.cube.fits.gz and the command line: **python3 pyHIIdet_ext_pipe3d.py NGC2906 NGC2906.Pipe3D.cube.fits.gz 3 1 2 4 45 245 2.3 1.0 0 0 1 3 1 1 2 file_for_save_results** The example cube is dataproduct of CALIFA survey for storage purposes. However, each script could be run it as standalone script.


