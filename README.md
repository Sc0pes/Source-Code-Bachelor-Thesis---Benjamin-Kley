Hello and welcome to my Bachelor Thesis source code!

All of the functions should be called from main, the most important functions will be create_data_set(), which takes multiple
tuples of the form (BooleanForEffect, EffectStrength). This and all the other functions become pretty self explanatory.
To be clear, I do not claim the code is perfect, or completely functional as published for that matter. However, this code is meant for exploration and to retrace as well as understand the
results of my thesis better. 

Important notes:
- The source code for the MATLAB code of Perrone et al. is available at: https://www.cvg.unibe.ch/media/project/perrone/tvdb/. However,
the images deconvolved are in the data-set/done/folderName/deconvolved/tv folder and can be used for comparison and of course for the 
recreation of the final results of my paper. The naming convention is: deblur_lambda_imageNumber_LambdaValue so e.g.: deblur_lambda_01_9e-07.fits = image number 1, lambda value: 9e-7.
- The same goes for BlurXTerminator: As the software is commercially available, the results used for the comparison in my thesis are available under
data-set/done/folderName/deconvolved/blur. The naming convention is: stellarDeconStrengthNonStellarDeconStrength so e.g.: 1640.fits = stellar strength = 0.16, non-stellar strength = 0.4.
- all images in the data-set are 32 bit float fits images
- The implementation for STARRED is available and should be functional, however, incase of problems visit: https://gitlab.com/cosmograil/starred , which is the gitab repo of the COSMOGRAIL team.

Short explanation for every class:
main.py:
Basically the control script from which every other function should be called.

WienerFilter.py:
Implementation of the WienerFilter and the inverse filter. The inverse filter hasn't been used in the bachelor thesis as it is of no use as soon 
as noise is introduced into an image, however, it is functional.

plot_metrics.py:
Basically just for plotting the results of different function calls and especially convergence functions. 

quantify.py:
All of the functions which measure quantifiable metrics such as SSIM, MSE and FWHM are implemented in this class.

Starred_Wavelet_Deconvolution.py:
Implementation of the STARRED algorithm. 

simulate_images.py:
This class implements all image degrading functions for PSF creation and spatially variant as well as invariant convolution functions.

Convergence.py:
Measures most of the convergence for each of the algorithms/outputs of the algorithms. 


Thank you! If you have any questions, please contact me at: benjamin.kley@web.de.

This repository was created by Benjamin Kley, student at the Eberhard Karls University of Tuebingen for the bachelor thesis on comparing deconvolution algorithms in astronomy.

















