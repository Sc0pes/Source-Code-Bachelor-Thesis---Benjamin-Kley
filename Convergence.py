from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from quantify import compute_mse
import numpy as np
from simulate_images import center_crop, plotPictures
import quantify as quant
from astropy.io import fits
import plot_metrics
import os

# helper function for loading fits image from memory
def load_fits_image(filepath):
    data = fits.getdata(filepath)
    #load fits image and convert if to gray values if it still rgb
    return quant.img_to_gray(data)

# test convergence for already created images from TV decon, images are saved in "output" directory with their lambdas for 
# determining which lambda produced the best results
def converge_TV(reference, blurredImage, imageSize, outputDirectory="outputs/deblur_lambda_*.fits"):
        # as reference is usually still 512x512, make sure it is 256x256 for comparison
        reference = center_crop(reference, imageSize)
        # get result dictionary from actualy compare function
        results = plot_metrics.compare_deblurs_TV(reference, outputDirectory)
        # get best TV deconvolved image and crop it to 256 (as the TV algorithm pads the image)
        decon_TV_Cropped = center_crop(load_fits_image(results["best_by_ssim"]['file']), 256)
        # plot results from comparison
        plotPictures([(reference, "Reference"),(center_crop(blurredImage,imageSize), "Blurred"),
                      (center_crop(decon_TV_Cropped, 128), f"TV Blind Deconvolution: best lambda: {results['best_by_ssim']['lambda']}"), 
                      (results["best_by_ssim"]["psf"], "PSF")])
        
# small helper needed for normalizing reference (if normalize = True in function call) + result in comparison
def norm_img(img):
        img = img.astype(np.float32)
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min + 1e-8)

#iterate until convergence function used for the RL and Wiener decon
def iterate_until_convergence(algorithm_name, algorithm_func, blurred, psf, reference,
                               param_range, max_steps=100000, stop_delta=0,
                               extra_args_func=None, image_size=256):

    #norm reference for comparison
    reference = norm_img(reference)
    results = []
    # define ssim for convergence stopping criteria
    prev_ssim = 0
    for step, param in enumerate(param_range):
        # if max steps have been reached without convergence, stop the algorithm and return results up until that point
        if step >= max_steps:
            print(f"{algorithm_name} reached max steps without convergence")
            break
        # leftover from original use, as it was supposed to test TV and BlurXTerminator convergence as well but turned out to be too complex in one function
        if extra_args_func:
            decon_result = algorithm_func(blurred, psf, param, *extra_args_func())
        else:
            # Wiener function
            if(algorithm_name == "Wiener Filter"):
                param = param / 10000
                decon_result = algorithm_func(blurred, psf, param)
            else:
                # RL function 
                decon_result = algorithm_func(blurred, psf, param)
                decon_result = center_crop(decon_result, 256)
                print(decon_result.shape)
        # crop images to same size for comparison
        result_norm = center_crop(decon_result, image_size)
        reference = center_crop(reference, image_size)
        # norm result as well
        result_norm = norm_img(result_norm)
        # compute reference metrics 
        mse_val = compute_mse(reference, result_norm)
        ssim_val = ssim(reference, result_norm, data_range=1.0)
        psnr_val = psnr(reference, result_norm, data_range=1.0)

        # add results to dictionary
        results.append((param, center_crop(decon_result, image_size), psnr_val, ssim_val, mse_val))
        # print intermediate steps 
        print(f"[{algorithm_name}] param={param} |  PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MSE={mse_val:.4e}")

        # Check convergence by SSIM improvement, if no further improvement = converged
        if step > 0:
            delta_ssim = ssim_val - prev_ssim
            if delta_ssim < stop_delta:
                print(f"[{algorithm_name}] Converged: SSIM={delta_ssim:.4e} < {stop_delta}")
                break
        
        prev_ssim = ssim_val

    # find best result by SSIM
    best = max(results, key=lambda x: x[3]) 
    # return all values in dictionary for final comparison with other algorithms
    return {
        'best_result': best[1],
        'best_param': best[0],
        'metrics': (best[2], best[3], best[4]),
        'all_results': results
    }

# this function has basically the exact same functionality as the decon_TV function - as some key features such as the n1/n2
# extraction are different, they have been separated to keep it simple
def converge_BLURX(reference, blurredImage, imageSize, outputDirectory="images-for-bachelor/blurxresults"):
    # norm + crop reference and blurred + noisy image
    reference = norm_img(center_crop(reference, imageSize))
    blurredImage = norm_img(center_crop(blurredImage, imageSize))

    results = plot_metrics.compare_deblurs_BLURX(
        reference,
        image_glob_pattern=os.path.join(outputDirectory, "*.fits"),
        psf_glob_pattern=None,     
        show_plots=True
    )

    # load best file and norm + crop it
    best_file = results["best_by_ssim"]["file"]
    decon_TV_Cropped = norm_img(center_crop(load_fits_image(best_file), imageSize))

    # extracts the n1/n2 label which are the stellar and non-stellar decon strengths of BlurXTerminator
    n1n2_label = results["best_by_ssim"].get("n1n2_label")
    if not n1n2_label:
        pair = results["best_by_ssim"].get("n1n2_pair")
        n1n2_label = f"{pair[0]:g}/{pair[1]:g}" if pair is not None else "unknown"

    # crates plots for final plitting
    plots = [
        (reference, "Reference"),
        (blurredImage, "Blurred"),
        (decon_TV_Cropped, f"BlurXTerminator: best decon settings n1/n2 = {n1n2_label}")
    ]

    plotPictures(plots)
