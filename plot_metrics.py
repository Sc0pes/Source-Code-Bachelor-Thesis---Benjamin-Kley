import matplotlib.pyplot as plt
import os, re, glob, numpy as np
from astropy.io import fits
from quantify import compute_mse as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# helper to norm images for comparison
def norm_img(img):
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min + 1e-8)

# center_crop as images have various different sizes from deconvolution (e.g., ref = 512x512/RL = 350x350/ Blur = 512x512..)
def center_crop(image, size):
    crop_h = size
    crop_w = size
    h, w = image.shape
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]

# plots metrics for the result dictionaries produced by the convergence functions
def plotMetrics(result):
    # Extract all metrics
    params = [r[0] for r in result['all_results']]
    psnr_vals = [r[2] for r in result['all_results']]
    ssim_vals = [r[3] for r in result['all_results']]
    mse_vals  = [r[4] for r in result['all_results']]

    # create  plots
    plt.figure(figsize=(15, 5))

    # Plot for PSNR
    plt.subplot(1, 3, 1)
    plt.plot(params, psnr_vals, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title('PSNR per iteration')
    plt.grid(True)

    # Plot for SSIM
    plt.subplot(1, 3, 2)
    plt.plot(params, ssim_vals, marker='o', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('SSIM')
    plt.title('SSIM per iteration')
    plt.grid(True)

    # Plot for MSE
    plt.subplot(1, 3, 3)
    plt.plot(params, mse_vals, marker='o', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE per iteration')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# again, helper to load fits image (as array this time)
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)

def extract_lambda(name):
    # complicated match to extract the lambda value from the image name
    m = re.search(r'_(\d+(?:\.\d+)?e[+-]?\d+)\.fits$', name)
    return float(m.group(1)) if m else None

def compare_deblurs_TV(
    reference,
    image_glob_pattern="outputs/deblur_lambda_*.fits",
    psf_glob_pattern="outputs/kernel_lambda_*.fits",
    show_plots=True
):
    # load reference image either vie filepath or cast to an array if the image is given to the function
    if isinstance(reference, str):
        ref = load_fits(reference)
    else:
        ref = np.asarray(reference, dtype=np.float64)

    # load the image files from memory
    img_files = sorted(glob.glob(image_glob_pattern), key=lambda s: (
        re.search(r'lambda_(\d+)', s) and int(re.search(r'lambda_(\d+)', s).group(1))
    ) or s)

    # load PSF kernels estimated by the TV algorithm from memory
    psf_files = sorted(glob.glob(psf_glob_pattern), key=lambda s: (
        re.search(r'lambda_(\d+)', s) and int(re.search(r'lambda_(\d+)', s).group(1))
    ) or s)

    # extract lambdas from image names
    lambdas = []
    parsed = []
    for f in img_files:
        lam = extract_lambda(f)
        lambdas.append(lam)
        parsed.append(lam is not None)

    # norm ground truth
    ref = norm_img(ref)

    # compute ssim,mse and psnr
    ssim_vals, mse_vals, psnr_vals = [], [], []
    images, psfs = [], []
    for i, (img_path, psf_path, lam) in enumerate(zip(img_files, psf_files, lambdas), start=1):
        img = load_fits(img_path)
        img = center_crop(img, ref.shape[0])
        img = norm_img(img)

        psf = load_fits(psf_path)

        images.append(img)
        psfs.append(psf)

        ssim_vals.append(ssim(ref, img, data_range=1.0))
        mse_vals.append(mse(ref, img))
        psnr_vals.append(psnr(ref, img, data_range=1.0))

    images = np.stack(images, axis=0)
    psfs = np.stack(psfs, axis=0)
    lambdas = np.asarray(lambdas, dtype=float)
    ssim_vals = np.asarray(ssim_vals, dtype=float)
    mse_vals = np.asarray(mse_vals, dtype=float)
    psnr_vals = np.asarray(psnr_vals, dtype=float)

    # best by ssim
    best_idx = int(np.argmax(ssim_vals))
    # values of best image as dictionary
    best = {
        "index": best_idx,
        "file": img_files[best_idx],
        "lambda": lambdas[best_idx],
        "ssim": ssim_vals[best_idx],
        "mse": mse_vals[best_idx],
        "psnr": psnr_vals[best_idx],
        "psf": psfs[best_idx], 
        "psf_file": psf_files[best_idx]
    }
    print("Best by SSIM:"
          f"idx={best_idx}, λ={best['lambda']:.3e}, SSIM={best['ssim']:.6f}, file={best['file']}")

    if show_plots:
        plt.figure(figsize=(15, 5))

        # PSNR
        plt.subplot(1, 3, 1)
        plt.plot(lambdas, psnr_vals, marker='o')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('PSNR')
        plt.title('PSNR per lambda')
        plt.grid(True, which='both', ls=':')

        # SSIM
        plt.subplot(1, 3, 2)
        plt.plot(lambdas, ssim_vals, marker='o', color='orange')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('SSIM')
        plt.title('SSIM per lambda')
        plt.grid(True, which='both', ls=':')

        # MSE
        plt.subplot(1, 3, 3)
        plt.plot(lambdas, mse_vals, marker='o', color='green')
        plt.xscale('log')
        plt.xlabel('Lambda')
        plt.ylabel('MSE')
        plt.title('MSE per lambda')
        plt.grid(True, which='both', ls=':')

        plt.tight_layout()
        plt.show()

    return {
        "files": img_files,
        "psf_files": psf_files,
        "lambdas": lambdas,
        "images": images,
        "psfs": psfs,
        "ssim": ssim_vals,
        "mse": mse_vals,
        "psnr": psnr_vals,
        "best_by_ssim": best,
    }





# basically the same function as compare_deblurs_TV, just for the images from BlurXTerminator
# copied for slight differences, for example the match for the n1/n2 pair in the name of the Blur image names
def compare_deblurs_BLURX(
    reference,
    image_glob_pattern="images-for-bachelor/blurxresults/*.fits",
    show_plots=True
):

    # helper to ectract the n1/n2 pair, which are the stellar and non-stellar decon strength
    def _extract_n1n2(path):
        base = os.path.basename(path)
        # complex match, bascially: any two numbers which are separated by something that is not a number
        nums = re.findall(r'(\d+(?:\.\d+)?)', base)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        # fallback
        m = re.search(r'(\d{4,})', base)
        if m:
            s = m.group(1)
            if len(s) % 2 == 0:
                return float(s[:len(s)//2]), float(s[len(s)//2:])
        return None

    # load reference image either vie filepath or cast to an array if the image is given to the function
    if isinstance(reference, str):
        ref = load_fits(reference)
    else:
        ref = np.asarray(reference, dtype=np.float64)
    # norm ground truth
    ref = norm_img(ref)

    # load image files and sort by n1/n2
    img_files = glob.glob(image_glob_pattern)

    def sort_key(p):
        pair = _extract_n1n2(p)
        return pair if pair is not None else (np.inf, np.inf, p)
    # actual sorting of the list
    img_files = sorted(img_files, key=sort_key)

    # use the n1/n2 numbers for the x-axis of the plot
    pairs = []
    labels = []
    for f in img_files:
        pr = _extract_n1n2(f)
        pairs.append(pr)
        # n1 = pr[0], n2 = pr[1]
        labels.append(f"{pr[0]:g}/{pr[1]:g}" if pr is not None else os.path.splitext(os.path.basename(f))[0])


    # compute ssim,mse and psnr
    ssim_vals, mse_vals, psnr_vals = [], [], []
    images, psfs = [], []
    for i, img_path in enumerate(img_files, start=1):
        img = load_fits(img_path)
        img = center_crop(img, ref.shape[0])
        img = norm_img(img)


        images.append(img)
        ssim_vals.append(ssim(ref, img, data_range=1.0))
        mse_vals.append(mse(ref, img))
        psnr_vals.append(psnr(ref, img, data_range=1.0))

    images = np.stack(images, axis=0)
    # cast to array
    ssim_vals = np.asarray(ssim_vals, dtype=float)
    mse_vals  = np.asarray(mse_vals, dtype=float)
    psnr_vals = np.asarray(psnr_vals, dtype=float)

    # best by SSIM
    best_idx = int(np.argmax(ssim_vals))
    # dictionaries with values of the best image
    best = {
        "index": best_idx,
        "file": img_files[best_idx],
        "ssim": float(ssim_vals[best_idx]),
        "mse": float(mse_vals[best_idx]),
        "psnr": float(psnr_vals[best_idx]),
        "psf": (psfs[best_idx] if psfs is not None else None),
        "n1n2_pair": pairs[best_idx],
        "n1n2_label": labels[best_idx],
    }
    print("Best by SSIM:"
          f"idx={best_idx}, n1/n2={best['n1n2_label']}, SSIM={best['ssim']:.6f}, file={best['file']}")

    if show_plots:
        x = np.arange(len(img_files))

        plt.figure(figsize=(15, 5))

        # PSNR
        plt.subplot(1, 3, 1)
        plt.plot(x, psnr_vals, marker='o')
        plt.xlabel('n1/n2 (stellar / non-stellar)')
        plt.ylabel('PSNR')
        plt.title('PSNR per n1/n2')
        plt.grid(True, which='both', ls=':')
        plt.xticks(x, labels, rotation=45, ha='right')

        # SSIM
        plt.subplot(1, 3, 2)
        plt.plot(x, ssim_vals, marker='o', color='orange')
        plt.xlabel('n1/n2 (stellar / non-stellar)')
        plt.ylabel('SSIM')
        plt.title('SSIM per n1/n2')
        plt.grid(True, which='both', ls=':')
        plt.xticks(x, labels, rotation=45, ha='right')

        # MSE
        plt.subplot(1, 3, 3)
        plt.plot(x, mse_vals, marker='o', color='green')
        plt.xlabel('n1/n2 (stellar / non-stellar)')
        plt.ylabel('MSE')
        plt.title('MSE per n1/n2')
        plt.grid(True, which='both', ls=':')
        plt.xticks(x, labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()
    # return entire dictionary for further processing
    return {
        "files": img_files,           
        "n1n2_pairs": pairs,            
        "n1n2_labels": labels,          
        "images": images,
        "psfs": psfs,
        "ssim": ssim_vals,
        "mse": mse_vals,
        "psnr": psnr_vals,
        "best_by_ssim": best,
    }
