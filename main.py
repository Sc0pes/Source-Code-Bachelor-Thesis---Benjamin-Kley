import numpy as np
import matplotlib.pyplot as plt
import quantify as quant
import WienerFilter as Wiener
from astropy.io import fits
from astropy.visualization import simple_norm
from skimage.restoration import richardson_lucy
from pathlib import Path
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_2dg 
from scipy.ndimage import shift
import os
import simulate_images
from Convergence import iterate_until_convergence
import plot_metrics
from simulate_images import center_crop, plotPictures, create_gaussian_psf
import json

deconAlgorithms = []

#list of tuples containing a raw image and their respective psf which has been extracted from pixinsight
image_psf_pairs = []

#leftover parameters for testing
iteration_RL = [500]
iteration_wiener = [0.0005]

#defines the amount of images showcasing original, blurred etc. that come before the actual deconvolved images
images_before_loop = 4

stars_cutouts = []

#predefined sigmas and seeds for convergence example
sigmas = [0.21, 0.18, 0.15, 0.25, 0.16, 0.13, 0.3, 0.22, 0.12, 0.17]
seeds = [42, 50, 99, 3, 27, 182, 88, 77, 12, 6]

raw_dir = Path("./images/raw")
hubble_dir = Path("./images/raw/hubble")

# extract psf from image, quite unreliable, especially for the blurred + noisy result, sometimes functional however
# Important: PSF size needs to be odd
# There are numerous checks for zero/nan values as this has been a major reason for failure
def extract_psf_from_image(image, psf_box_size=25, min_fwhm_for_psf=3.0, threshold_sigma_psf=3.0, num_psf_stars=15):
    # check for nans in the input image
    if np.any(np.isnan(image)):
        #Attempt to replace nans with median/mean or return a default PSF
        image = np.nan_to_num(image, nan=np.nanmedian(image)) # Replace nans with median for robustness
        print("Nans in input image, replaced with median for extraction")

    # detect stars within image, initial sigma guess of 3.0
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    #check if extracted data from clipping is nan
    if np.isnan(std) or std == 0:
        print("std = nan or zero, can't detect stars, creating synthetic gaussian psf")
        # create gaussian psf as fallback
        return create_gaussian_psf(psf_box_size, 5.0) 

    #extract fitting stars with initial guesses for values
    daofind = DAOStarFinder(fwhm=5, threshold=threshold_sigma_psf * std,
                            sharplo=0.0, sharphi=2.0, roundlo=-1.0, roundhi=1.0)
    
    #subtract background from image (similar to fwhm extraction in quantify)
    bkg_sub = image - median 
    bkg_sub[bkg_sub < 0] = 0 
    sources = daofind(bkg_sub)

    #if there are less than 5 stars, calculate a synthetic gaussian psf
    if sources is None or len(sources) < 5:
        print("Not enough stars found, creating synthetic, gaussian psf")
        #create synthetic psf
        return create_gaussian_psf(psf_box_size, 5.0)

    #sort stars depending on brightness
    sources.sort('flux', reverse=True)
    psf_stars_cutouts = []
    
    #determine amount of stars to process
    stars_to_process = min(num_psf_stars, len(sources))

    #loop through the found stars and determine their position within the image
    for k, star in enumerate(sources[:stars_to_process]):
        x_centroid, y_centroid = star['xcentroid'], star['ycentroid']
        print(f"  Processing star {k+1}/{stars_to_process} at ({x_centroid:.1f}, {y_centroid:.1f})")
        #create cutout of the current star, similar to fwhm extraction
        half_box = psf_box_size // 2
        min_x = int(x_centroid - half_box)
        max_x = int(x_centroid + half_box + 1)
        min_y = int(y_centroid - half_box)
        max_y = int(y_centroid + half_box + 1)

        #if star is too close to the image border, ignore that star
        if not (min_x >= 0 and max_x <= image.shape[1] and 
                min_y >= 0 and max_y <= image.shape[0]):
            print(f"Star {k+1} skipped = too close to edge")
            continue 

        try:
            #create cutout of found star
            cutout = Cutout2D(image, (x_centroid, y_centroid), psf_box_size, mode='strict')
            #check if any of the values within the cutout are nans or zero + skip that star
            if cutout.data.size == 0 or np.all(cutout.data == 0) or np.any(np.isnan(cutout.data)):
                continue
            
            #determine background of star cutouts
            cutout_mean, cutout_median, cutout_std = sigma_clipped_stats(cutout.data, sigma=3.0)
            #check again if there are any nans in the cutout and skip if thats the case
            if np.isnan(cutout_median) or np.isnan(cutout_std):
                continue
            
            #subtract background from star cutout
            bkg_subtracted_cutout = cutout.data - cutout_median
            bkg_subtracted_cutout[bkg_subtracted_cutout < 0] = 0

            #check if any of the values are nan after background extraction and skip if thats the case
            if np.any(np.isnan(bkg_subtracted_cutout)):
                continue
            
            #check if sum of all the pixels is < 0 to make sure there is a star there, if not, skip
            if bkg_subtracted_cutout.sum() <= 0:
                continue

            #center image around the stars center 
            cx, cy = centroid_2dg(bkg_subtracted_cutout)
            # check again for nans..
            if np.isnan(cx) or np.isnan(cy): 
                continue
            
            #determine how far star is away from star cutout center
            shift_x = ((psf_box_size - 1) / 2) - cx
            shift_y = ((psf_box_size - 1) / 2) - cy

            #shift image cutout so it is centered around star
            recentered_cutout = shift(bkg_subtracted_cutout, (shift_y, shift_x), order=3, mode='constant', cval=0)
            
            #check if the star cutout is the same size as psf box size, if it isn't (=probably too close to edge), then skip
            if recentered_cutout.shape != (psf_box_size, psf_box_size):
                continue
            #check if any values of the recentered cutouts are nans
            if np.any(np.isnan(recentered_cutout)):
                continue

            #check if the sum of all the values in the cutout is > 0, meaning there is a star there to be extracted
            current_cutout_sum = recentered_cutout.sum()
            if current_cutout_sum <= 0: 
                continue
            
            #normalize the cutout
            recentered_cutout /= current_cutout_sum

            #make list of cutouts
            psf_stars_cutouts.append(recentered_cutout)

        except Exception as e:
            print(f"error processing star {k+1}")
            continue

    #if there couldn't be any cutouts made, return error and create synthetic psf
    if not psf_stars_cutouts:
        print("error: No valid star cutouts collected for psf extraction")
        print("returning synthetic gaussian psf instead")
        return create_gaussian_psf(psf_box_size, 5.0)

    #check if cutouts all have the same shape (check for safety, but has already been checked earlier)
    # return synthetic psf if not all of the same shape
    if not all(c.shape == psf_stars_cutouts[0].shape for c in psf_stars_cutouts):
        return create_gaussian_psf(psf_box_size, 5.0)

    #stack all psf star cutouts for accurate estimate across image
    stacked_psf = np.median(psf_stars_cutouts, axis=0)

    # check created PSF for nan and return synthetic PSF if so
    if np.any(np.isnan(stacked_psf)):
        return create_gaussian_psf(psf_box_size, 5.0)

    #sum psf values for normalization
    final_psf_sum = stacked_psf.sum()

    #check if psf values are > 0 and if psf size > 0, if not, again, synthetic psf
    if final_psf_sum <= 0 or stacked_psf.size == 0:
        return create_gaussian_psf(psf_box_size, 5.0)
    
    #normalize psf
    stacked_psf /= final_psf_sum
    
    #final check for nans, or if size is 0 and return synthetic psf as last option
    if np.any(np.isnan(stacked_psf)) or stacked_psf.size == 0:
        return create_gaussian_psf(psf_box_size, 5.0)

    return stacked_psf  

# helper for image normalization (again..)
def normalize_image(img):
        img = img.astype(np.float32)
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)

def load_fits_image(filepath):
    data = fits.getdata(filepath)
    #load fits image and convert if to gray values if it still rgb
    return quant.img_to_gray(data)

#load all images in hubble directory for deconvolving all in deconAndShow() (leftover from original plan, kept since it is useful for some cases)
def loadImages():
    #load images from hubble directory into list
    for image_path in hubble_dir.glob("*.fits"):
        # originally the plan was to extract the psf in pixinsight and store it with the image in a list. We now create the psf 
        # within this program so we store a placeholder for now incase we do need a manual psf from pixinsight to create better images
        image_psf_pairs.append(image_path)

# save raw fits mostly used for deconvolving images with TV and BlurXTerminator
def save_raw_fits(images, save_dir='./images-for-bachelor'):
    # make sure directory exists
    os.makedirs(save_dir, exist_ok=True)
    # loop through images and save as contiguousarray (has to be for fits)
    for i, img in enumerate(images):
        x = img
        x = np.ascontiguousarray(x)
        # create generic file name
        filename = os.path.join(save_dir, f'image_{i:03}.fits')
        fits.PrimaryHDU(x).writeto(filename, overwrite=True)

# plotPictures for PSFs, technically the plotPictures in simulateImages does the same but this is kept for compatibility
def plotPictures_hot(images):
    stretch = 'sqrt'
    percent = 99
    images_per_row = 5
    num_images = len(images)
    
    # norm all images by same amount
    stack = np.hstack([img.ravel() for img in images])
    norm = simple_norm(stack, stretch, percent=percent)

    # determine rows and columnds needed for plotting
    rows = (num_images + images_per_row - 1) // images_per_row
    cols = min(images_per_row, num_images)

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.atleast_2d(axs).reshape(-1)

    #loop through psfs and create titles
    for idx, img in enumerate(images):
        axs[idx].imshow(img, cmap='hot', origin='lower', norm=norm)
        axs[idx].set_title(f"image {idx}")
        axs[idx].axis('off')

    # hide any subplots without images
    for k in range(len(images), len(axs)):
        axs[k].axis('off')

    plt.tight_layout()
    plt.show()

# function which takes an image and plots the image degraded with a bunch of different degradations
# used for tests and 
def testImages(image):
    print("blurred")
    blurred, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.5), coma=(False,0.1), trackingError=(False,0.25), defocus=(False, 0.05))
    imgs = [pair[0] for pair in random_imgs_and_psfs]
    psfs = [pair[1] for pair in random_imgs_and_psfs]
    plotPictures(imgs)
    plotPictures_hot(psfs)
    # various prints to see if the function is still runing and which image degradation takes longest
    # this is especially useful for the spatially variant PSF convolution, as it is incredibly inefficient
    print("blurred1")
    blurred1, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(False,1), trackingError=(True,0.25), defocus=(False, 0.05))
    print("blurred2")
    blurred2, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(False,1), trackingError=(False,0.25), defocus=(True, 0.05))
    print("blurred3")
    blurred3, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(False,1), trackingError=(False,0.25), defocus=(False, 0.05))
    print("blurred4")
    blurred4, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(True,0.1), trackingError=(True,0.25), defocus=(False, 0.05))
    print("blurred5")
    blurred5, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(True,0.1), trackingError=(True,0.25), defocus=(True, 0.05))
    print("blurred6")
    blurred6, random_imgs_and_psfs, stacked_psf, sigma_values, seed_values = create_dataset(image, noiseIndB=(True, 30), atmosphere=(True, 0.3), coma=(True,0.1), trackingError=(True,0.25), defocus=(False, 0.05))
    plotPictures([blurred, blurred1, blurred2, blurred3, blurred4, blurred5, blurred6])

# returns an image (or images incase the atmospheric degradation = True), based on the tuples given to the function
def create_dataset(image, psf_shape=(25,25), noise_in_dB=(True, 30), atmosphere=(True, 1), coma=(False,1), trackingError=(False,1), defocus=(False, 1)):
    blurred, imagesAndPsfs, stacked_psf, sigma_values, seed_values = [],[],[],[],[]
    # if atmosphere = True, a sigma range for the Gaussian PSFs will be determined
    if(atmosphere[0]):    
        if(atmosphere[1] >= 0.2):
            range = (atmosphere[1] - 0.2, atmosphere[1])
        else:
            range = (0, atmosphere[1])
        blurred, imagesAndPsfs, stacked_psf, sigma_values, seed_values = simulate_images.simulate_atmospheric_stack(image, num_frames=10, psf_shape=psf_shape, sigma_range=range)
    # if coma = true, run spatially variant convolution
    if(coma[0]):
            if(len(blurred) < 1):
                blurred = image
            blurred = simulate_images.spatially_variant_convolution(
                            blurred,
                            psf_size=psf_shape[0],
                            img_degrading_func=lambda x, y, **kwargs: simulate_images.coma_psf(
                                x, y,
                                shape=psf_shape,
                                center=(256, 256),
                                strength=coma[1],        # tail length (standard = 0.2)
                                sigma= coma[1] / 4,          # tightness of the core (standard = 0.05)
                                tail_weight=coma[1] + 0.1     # tail intensity (standard = 0.3)
                            )
                        )
            
    if(trackingError[0]): 
            trackingPSF = simulate_images.tracking_error_psf(
                                shape=psf_shape,
                                length=trackingError[1] * 10, # length of line representing tracking error (standard = 5)
                                angle_deg=45  # direction of error (standard = 45)
                            )
            blurred = simulate_images.convolve_with_psf(
                            blurred,
                            trackingPSF
                        )
            if(len(stacked_psf) > 1):
                denom = 2
            else: 
                denom = 1
            stacked_psf += trackingPSF
            stacked_psf = stacked_psf / denom
            if np.sum(stacked_psf) > 0:
                stacked_psf /= np.sum(stacked_psf)

    if(defocus[0]):
            defocusPSF = simulate_images.defocus_psf(
                                shape=psf_shape,
                                sigma=defocus[1] * 0.5  # amount of blurriness (standard 0.2)
                            )
            blurred = simulate_images.convolve_with_psf(
                            blurred,
                            defocusPSF
                        )
            if(len(stacked_psf) > 1):
                denom = 2
                stacked_psf += defocusPSF
                stacked_psf = stacked_psf / denom
            else: 
                stacked_psf = defocusPSF

            if np.sum(stacked_psf) > 0:
                stacked_psf /= np.sum(stacked_psf)
    # if there is no PSF available (the case if only coma was added to the image), extract a PSF from the degraded image
    if(len(stacked_psf) < 1):
        stacked_psf = extract_psf_from_image(blurred)
    # add noise to the degraded image
    if(noise_in_dB[0]):
        blurred = simulate_images.add_white_gaussian_noise(blurred, noise_in_dB[1])
    return blurred, imagesAndPsfs, stacked_psf, sigma_values, seed_values

#leftover from ealier testing, might still be useful
deconAlgorithms.append(["Richardson Lucy", richardson_lucy, iteration_RL])
deconAlgorithms.append(["Wiener Filter", Wiener.wiener_deconvolution, iteration_wiener])
deconAlgorithms.append(["Inverse Filter", Wiener.inverse_filter, iteration_wiener])


def decon_and_show(image_size):
#for every image we have in the hubble directory 
    for image_path in image_psf_pairs:     
        stringzzz = ["Galaxy_Gaussian-5_PSF25_512","Galaxy_Gaussian-5_Noise-20_PSF25_512","Galaxy_Gaussian-5_Noise-40_PSF25_512","Galaxy_Gaussian-5_Noise-60_PSF25_512","Galaxy_Noise-40Atmosphere-0103_PSF25_512", 
                     "Galaxy_Noise-40Atmosphere-0103_Tracking-3_NOPSF25_512", "Galaxy_Noise-40Atmosphere-0103_Coma-02_Defocus-015_NOPSF25_512","Galaxy_Noise-40Atmosphere-0103_Coma-03_NOPSF25_512", 
                     "Galaxy_Noise-40Atmosphere-0103_Defocus-03_NOPSF25_512","M42_Noise-40Atmosphere-0103_NOPSF25_512", "Jupiter_Noise-40Atmosphere-0103_Defocus-05_NOPSF25_512",
                     "Jupiter_Noise-40Atmosphere-0103_NOPSF25_512","M42_Noise-40_Coma-02_NOPSF25_512"]
        for i in range(14):
                image_name = stringzzz[i]
            
                decon_directory = "./data-set/done/" + image_name
                image_directory = "/" + image_name
                result_directory = decon_directory + "/results"
                # load images from the degradation process, includes reference, degraded image, given PSF and estimated PSF
                image = load_fits_image(decon_directory + image_directory + "_ref.fits")
                blurred = load_fits_image(decon_directory + image_directory + "_blur.fits")
                psf_stack = load_fits_image(decon_directory + image_directory+ "_psf.fits")
                est_psf = load_fits_image(decon_directory + image_directory + "_extracted.fits")

                reference = center_crop(image, 256)
                #crop it to 350 to avoid edge artefacts with RL and Wiener
                blurred_rl = center_crop(blurred, 350)
                blurred = center_crop(blurred, 256)

                #CONVERGENCE OF ALL ALGORITHMS
                #TV
                results_TV = plot_metrics.compare_deblurs_TV(reference, decon_directory + "/deconvolved/tv/deblur_lambda_*.fits")
                #BLURX
                results_BLURX = plot_metrics.compare_deblurs_BLURX(reference, decon_directory + "/deconvolved/blur/*.fits")
                #WIENER
                results_Wiener = iterate_until_convergence(
                                algorithm_name="Wiener Filter",
                                algorithm_func=Wiener.wiener_deconvolution,
                                blurred=blurred_rl,
                                psf=psf_stack,
                                reference=reference,
                                param_range=range(1, 100000),
                                stop_delta=0,
                                image_size=image_size
                            )
                results_Wiener_est = iterate_until_convergence(
                                algorithm_name="Wiener Filter",
                                algorithm_func=Wiener.wiener_deconvolution,
                                blurred=blurred_rl,
                                psf=est_psf,
                                reference=reference,
                                param_range=range(1, 100000),
                                stop_delta=0,
                                image_size=image_size
                            )
                #RL
                results_RL = iterate_until_convergence( 
                                algorithm_name="Richardson-Lucy",
                                algorithm_func=richardson_lucy,
                                blurred=blurred_rl,
                                psf=psf_stack,
                                reference=reference,
                                param_range=range(1, 4000),
                                stop_delta=1e-5,
                                image_size=image_size
                            )
                results_RL_est = iterate_until_convergence( 
                                algorithm_name="Richardson-Lucy",
                                algorithm_func=richardson_lucy,
                                blurred=blurred_rl,
                                psf=est_psf,
                                reference=reference,
                                param_range=range(1, 4000),
                                stop_delta=1e-5,
                                image_size=image_size
                            )

                #COMPARISON
                #compare_all(reference, blurred, results_TV, results_BLURX, results_RL, results_Wiener, results_RL_est, results_Wiener_est, result_directory, image_name)

# run program
loadImages()
decon_and_show(256)




