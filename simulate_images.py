from astropy.modeling import models
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from skimage.draw import line as draw_line
from astropy.visualization import simple_norm
import quantify as quant 

#shows a single image stretched in grayscale
def showImage(image):
    #if image isn't empty, turn image to grayscale and show stretched
    if(image.shape[0] > 1):
        image = quant.img_to_gray(image)
    #display with sqrt stretch and 99% clip
    plt.imshow(image, cmap='gray', origin='lower', norm=simple_norm(image, 'sqrt', percent=99))
    plt.show()


#expects an Array of the form: [imageArray, "titleOfImage"]
def plotPictures(images_with_titles):
    stretch = 'sqrt'
    percent = 99
    images_per_row = 5
    num_images = len(images_with_titles)

    #computes plot size
    rows = (num_images + images_per_row - 1) // images_per_row
    cols = min(images_per_row, num_images)

    #create plot
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = np.atleast_2d(axs).reshape(-1)

    #iterates through images and if the name of the image contains "psf", turns plot to "hot", otherwise gray
    for idx, (img, title) in enumerate(images_with_titles):
        cmap = 'hot' if "psf" in title.lower() else 'gray'
        norm = simple_norm(img, stretch, percent=percent)
        axs[idx].imshow(img, cmap=cmap, origin='lower', norm=norm)
        axs[idx].set_title(title)
        axs[idx].axis('off')

    #hide unused plots
    for k in range(len(images_with_titles), len(axs)):
        axs[k].axis('off')

    plt.tight_layout()
    plt.show()


#returns a centered square crop of size `size`×`size`
def center_crop(image, size):
    crop_h = size
    crop_w = size
    h, w = image.shape
    #compute top left corner of crop so it is centered
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    #splice image array and return crop
    return image[start_y:start_y+crop_h, start_x:start_x+crop_w]


#adds white Gaussian noise to the image for a specified SNR (in dB)
#optionally returns a noise map for the STARRED algorithm
def add_white_gaussian_noise(image, snr_db, return_noise_map=False):
    #convert SNR from dB to normal
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power)

    #generate randomly Gaussian noise and add it to imgae
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise

    #return noisy image + optional noise map
    if return_noise_map:
        noise_map = np.full_like(image, sigma, dtype=np.float32)
        return noisy_image.astype(np.float32), noise_map
    else:
        return noisy_image.astype(np.float32)


#applies a spatially variant convolution where the PSF depends on (x,y)
#img_degrading_func(x, y, shape, center) return a PSF for that location
def spatially_variant_convolution(image, psf_size, img_degrading_func):
    h, w = image.shape
    pad = psf_size // 2
    #pad reflectively so borders don't show artefacts
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)

    #loop over every pixel and convolve with variant PSF
    for i in range(h):
        for j in range(w):
            #splice padded array for patch with size of psf
            patch = padded[i:i + psf_size, j:j + psf_size]
            #mirror coordinates relative to image center
            mirrored_x = w - j - 1
            mirrored_y = h - i - 1

            #generate and normalize PSF
            psf = img_degrading_func(mirrored_x, mirrored_y, shape=(psf_size, psf_size), center=(w // 2, h // 2))
            psf /= psf.sum()
            #convolve patch by using a weighted sum
            output[i, j] = np.sum(patch * psf)

    return output


#builds a random elliptical Gaussian PSF (rough atmospheric model)
#ellipticity and rotation are randomized + seeded for reproducability
def atmospheric_psf(shape=(25, 25), base_sigma=1.0, ellipticity_range=(0.0, 0.15), seed=None):
    h, w = shape
    #generate seed 
    rng = np.random.default_rng(seed)
    ellipticity = rng.uniform(*ellipticity_range)
    angle = rng.uniform(0, 2 * np.pi)

    #coordinate grid and rotation
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    x_rot = np.cos(angle) * x + np.sin(angle) * y
    y_rot = -np.sin(angle) * x + np.cos(angle) * y

    #different elliptical spreads along axes
    sigma_x = base_sigma * (1 + ellipticity)
    sigma_y = base_sigma * (1 - ellipticity)

    #normalize elliptical Gaussian
    psf = np.exp(-((x_rot / sigma_x)**2 + (y_rot / sigma_y)**2) / 2)
    psf /= psf.sum()
    return psf


#simulates an image stack by creating multiple PSFs and averaging the final image
#returns: (stacked_image, [(blur_i, psf_i)...], stacked_psf, used_sigmas, used_seeds)
def simulate_atmospheric_stack(image, num_frames=10, psf_shape=(21, 21), sigma_range=(0.1, 0.3), 
                               seed=None, predefined_sigmas=None, predefined_seeds=None):
    rng = np.random.default_rng(seed)

    #checks if amount of sigmas matches the amount of images
    if predefined_sigmas is not None and len(predefined_sigmas) != num_frames:
        raise ValueError("Length of predefined_sigmas must match num_frames")
    if predefined_seeds is not None and len(predefined_seeds) != num_frames:
        raise ValueError("Length of predefined_seeds must match num_frames")

    #empty arrays for processing and storing returns 
    stack = np.zeros_like(image, dtype=np.float32)
    psf_stack = np.zeros(psf_shape, dtype=np.float32)
    random_imgs_and_psfs = []
    sigma_values = []
    seed_values = []

    #simulate single exposures
    for i in range(num_frames):
        #pick random or predefined sigma
        if predefined_sigmas is not None:
            sigma_i = predefined_sigmas[i]
        else:
            sigma_i = rng.uniform(*sigma_range)

        #pick random or predefined seed
        if predefined_seeds is not None:
            seed_i = predefined_seeds[i]
            local_rng = np.random.default_rng(seed_i)
        else:
            seed_i = rng.integers(0, 1e9)
            local_rng = np.random.default_rng(seed_i)

        #create PSF, convolve image, and add to image
        psf_i = atmospheric_psf(shape=psf_shape, base_sigma=sigma_i, seed=local_rng)
        blurred_i = convolve_with_psf(image, psf_i)

        random_imgs_and_psfs.append((blurred_i, psf_i))
        stack += blurred_i
        psf_stack += psf_i
        sigma_values.append(sigma_i)
        seed_values.append(seed_i)

    #average image and PSF, normalize PSF again just incase
    stacked_image = stack / num_frames
    stacked_psf = psf_stack / num_frames
    if np.sum(stacked_psf) > 0:
        stacked_psf /= np.sum(stacked_psf)

    return stacked_image, np.array(random_imgs_and_psfs, dtype=object), stacked_psf, sigma_values, seed_values


#symmetric Gaussian PSF used to simulate simple defocus (could technically use create_gaussian_psf() instead..)
def defocus_psf(shape=(25, 25), sigma=0.3):
    h, w = shape
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return psf / psf.sum()


#PSF for tracking error: a short line with `length` rotated by `angle_deg`
def tracking_error_psf(shape=(25, 25), length=9, angle_deg=0):
    h, w = shape
    psf = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    angle_rad = np.deg2rad(angle_deg)

    #vector along rotation angle
    dx = int((length / 2) * np.cos(angle_rad))
    dy = int((length / 2) * np.sin(angle_rad))

    #draw symmetric line across the center
    rr, cc = draw_line(cy - dy, cx - dx, cy + dy, cx + dx)
    #clip to edges
    valid = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
    psf[rr[valid], cc[valid]] = 1.0
    #normalize
    psf /= psf.sum()
    return psf


#visualizes grid of PSFs produced at different locations in the image, created mainly for showung coma_psf()
def visualize_psf_grid(image_shape=(512, 512), grid_shape=(4, 4), psf_shape=(41, 41), degrading_function=None):
    h, w = image_shape
    rows, cols = grid_shape
    #grid of representative points 
    cx = np.linspace(0, w, cols, endpoint=False) + w // (2 * cols)
    cy = np.linspace(0, h, rows, endpoint=False) + h // (2 * rows)

    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for i, y in enumerate(cy):
        for j, x in enumerate(cx):
            #mirror coordinates (has to match spatially_variant_convolution())
            mirrored_x = w - int(x) - 1
            mirrored_y = h - int(y) - 1

            #create PSF and display it 'hot' in the 'plot'
            psf = degrading_function(mirrored_x, mirrored_y, shape=psf_shape, center=(w // 2, h // 2))
            ax = axs[i, cols - j - 1]  #flip x so figure matches image coordinates
            ax.imshow(psf, cmap='hot', origin='lower')
            ax.set_title(f"PSF at ({int(x)}, {int(y)})")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


#coma-like PSF with gaussian symmetric core and shifted gaussian as tail
def coma_psf(x, y, shape=(25, 25), center=(1280, 1280), strength=0.1, sigma=0.07, tail_weight=1):
    h, w = shape
    #normalize grid to [-1,1] so sigma/tail are not dependent on resolution of image
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

    #Direction vector from center
    dx = x - center[0]
    dy = y - center[1]
    norm = np.hypot(dx, dy) + 1e-6
    ux = dx / norm
    uy = dy / norm

    #Strength of shift gets bigger the further away from the center the pixel position is
    max_radius = np.hypot(center[0], center[1])
    scaling = strength * (norm / max_radius)

    #offset for tail
    tail_shift_x = scaling * ux * 1.5
    tail_shift_y = scaling * uy * 1.5

    #core Gaussian, centered
    core = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    #tail Gaussian, broader at the tail
    tail = np.exp(-((xx - tail_shift_x)**2 + (yy - tail_shift_y)**2) / (2 * (2 * sigma)**2))

    #Blend core and tail + normalize
    psf = (1 - tail_weight) * core + tail_weight * tail
    psf /= psf.sum()
    return psf


#convolves a 2-D image with a given PSF using Fourier Transform for faster compute
def convolve_with_psf(img, psf):
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale, got shape {img.shape}")

    #make sure PSF.sum() = 1 
    psf = psf.astype(np.float32)
    psf /= psf.sum() if psf.sum() else 1.0 

    #FFT convolution
    blurred = fftconvolve(img, psf, mode="same")
    return blurred.astype(img.dtype, copy=False)


#creates a centered Gaussian PSF from FWHM value, multiple checks for nans in image array because of various failures
def create_gaussian_psf(size, fwhm_estimate):
    #convert FWHM to sigma for gaussian creation
    sigma = fwhm_estimate / 2.355
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    g_model = models.Gaussian2D(amplitude=1.0,
                                x_mean=(size - 1) / 2,
                                y_mean=(size - 1) / 2,
                                x_stddev=sigma,
                                y_stddev=sigma)
    gauss_psf = g_model(x_grid, y_grid)
    
    #check for nans in PSF
    if np.any(np.isnan(gauss_psf)):
        print("gaussian psf contains nans")
        return np.ones((size, size)) / (size*size) #returns default psf
    
    #normalize, if not normalized, create PSF consisting of only 1s
    if gauss_psf.sum() > 0:
        gauss_psf /= gauss_psf.sum()
    else:
        print("gaussian sum is zero, returning 1s array")
        return np.ones((size, size)) / (size*size) 
    
    #check for nans after normalization
    if np.any(np.isnan(gauss_psf)):
        print("normalized gauss psf contains nans")
        return np.ones((size, size)) / (size*size) 
    
    return gauss_psf


#blur image with defined fwhm, basically just for convenience and easier name
def addAtmosphericBlur(image, fwhm_blur):
    psf_for_blur = create_gaussian_psf(25, fwhm_blur)
    return convolve_with_psf(image, psf_for_blur)