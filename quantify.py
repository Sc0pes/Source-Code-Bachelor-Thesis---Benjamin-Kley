import numpy as np
from skimage.metrics import structural_similarity as ssim  
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D

#compute the fwhm of stars in the image, not needed for any of the calcs but used as 
# indicator as to how much image has been deconvolved
# unreliable for blurred and noisy images as they don't allow for reliable star detection
def compute_fwhm(image, box_size=25, threshold_sigma=4.0): 
    # clip outliers from image, median = background/sky, std = noise
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    #find suitable stars for extraction
    daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma * std,
                            sharplo=0.2, sharphi=1.0, roundlo=-0.5, roundhi=0.5)
    
    #subtract background from image
    sources = daofind(image - median) 
    
    #if no suitable stars found, return nan
    if sources is None or len(sources) == 0:
        print("No stars found")
        return np.nan

    #sort stars by brightness
    fwhms = []
    sources.sort('flux', reverse=True) 
    
    for star in sources[:10]:
        # get coordinates for every star, if they are in bounds, continue otherwise skip
        x, y = star['xcentroid'], star['ycentroid']
        if not (0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]):
            continue

        #make cutout centered around star
        try:
            cutout = Cutout2D(image, (x, y), box_size, mode='partial')
            # safety check because of previous failures = skip stars/cutouts that aren't correct
            if cutout.data.size == 0 or np.all(cutout.data == 0) or np.all(np.isnan(cutout.data)):
                continue

            #determine background and subtract it from the cutout
            cutout_mean, cutout_median, cutout_std = sigma_clipped_stats(cutout.data, sigma=3.0)
            cutout_data_bkg_subtracted = cutout.data - cutout_median

            #determine size of cutout 
            y_grid, x_grid = np.mgrid[:cutout.data.shape[0], :cutout.data.shape[1]]
            
            #initial sigma guess of the width of fwhm with gaussian function
            initial_sigma_guess = star['fwhm'] / 2.355 if 'fwhm' in star.colnames and star['fwhm'] > 0 else 5.0 / 2.355
            
            #initial guess of the gaussian peak value
            amplitude_guess = cutout_data_bkg_subtracted.max()
            if amplitude_guess <= 0: 
                continue
            #make gaussian model based on amplitude guess, cutout shape and initial width guess
            g_init = models.Gaussian2D(amplitude=amplitude_guess,
                                       x_mean=cutout.data.shape[1] / 2,
                                       y_mean=cutout.data.shape[0] / 2,
                                       x_stddev=initial_sigma_guess,
                                       y_stddev=initial_sigma_guess)
            
            #constrains on model, width of curve bigger than 0.5 sigma and amplitude has to be > 0
            g_init.x_stddev.min = 0.5 
            g_init.y_stddev.min = 0.5
            g_init.amplitude.min = 0.0 

            #fit model
            fit_p = fitting.LevMarLSQFitter()
            
            g_fit = fit_p(g_init, x_grid, y_grid, cutout_data_bkg_subtracted)

            #rejects the model if any of the values are below minimums for a good star 
            if (np.abs(g_fit.x_stddev.value) < 0.5 or np.abs(g_fit.y_stddev.value) < 0.5 or 
                np.abs(g_fit.x_stddev.value) > box_size / 2 or np.abs(g_fit.y_stddev.value) > box_size / 2):
                continue
            
            #calculate fwhm values in x and y direction and average it
            fwhm_x = 2.355 * np.abs(g_fit.x_stddev.value)
            fwhm_y = 2.355 * np.abs(g_fit.y_stddev.value)
            fwhm = (fwhm_x + fwhm_y) / 2
            
            #if the value makes sense add it to fwhm list
            if 1.0 <= fwhm <= box_size: 
                fwhms.append(fwhm)
        #safety check for fwhm because of previous failures
        except Exception as e:
            print("FWHM fitting error for stars")
            continue
    # return median of fwhm list for representative value
    if fwhms:
        return np.median(fwhms)
    # if no values are in fwhm, return nan
    else:
        print("No FWHM measurement possible")
        return np.nan 

# helper function, turns image to gray-scale if it still 3-channel
def img_to_gray(image):
    return np.mean(image, axis=0) if image.ndim == 3 else image

def compute_mse(original, decon_image):
    # MSE between original and deconvolved image
    #closer to 0 = better (0 == identical image)

    #safety check because of multiple errors..
    if original.shape != decon_image.shape:
        raise ValueError("Images must have the same size")
    # make sure they are both float32 (should be the case already anyway)
    ref  = original.astype(np.float32)
    dec = decon_image.astype(np.float32)
    return np.mean((ref - dec) ** 2)

def compute_ssim(original, decon_image):
    # Structural Similarity Index which compares not only pixel values but structures within the image  
    # range from -1 to 1 (1 == identical image)
    # Convert to gray if not already done
    orig_gray  = img_to_gray((original).astype(np.float32))
    decon_gray = img_to_gray((decon_image).astype(np.float32))
    # define data range from max/min intensity values
    data_range = orig_gray.max() - orig_gray.min() or 1
    return ssim(orig_gray, decon_gray, data_range=data_range)
