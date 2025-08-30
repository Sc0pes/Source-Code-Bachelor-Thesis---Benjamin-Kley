import numpy as np
from astropy.stats import sigma_clipped_stats

# estimates the noise variance across the image
def estimate_noise_variance(image):
    #use sigma_clipped_stats to only take the background into account for noise variance
    #and ignore other objects (e.g., stars)
    _, _, std = sigma_clipped_stats(image, sigma=3.0)
    return std**2


# estimates the total signal variance across the whole image
# used together with noise variance to get a rough SNR value
def estimate_signal_variance(image):
    return np.var(image)


def estimateBalance(image):
    # estimate Wiener balance parameter as signal_var / noise_var for rough SNR
    # not really relevant for the use of this paper, since 10 000 different balance factors are used
    # for convergence, but a nice to have regardless
    noise_var = estimate_noise_variance(image)
    signal_var = estimate_signal_variance(image)
    return signal_var / noise_var


def inverse_filter(image, psf):
    # inverse filtering in the frequency domain - no noise handling
    # Convert image and PSF to frequency domain, PSF is padded to image size
    G = np.fft.fft2(image)
    H = np.fft.fft2(psf, s=image.shape)

    # complex conjugate of psf + take absolute of psf and then magnitude
    H_conj = np.conj(H)
    H_mag = np.abs(H) ** 2

    # actual inverse filtering step
    F_hat = G * H_conj / (H_mag)

    # transform back to spatial domain 
    result = np.abs(np.fft.ifft2(F_hat))
    return result


def wiener_deconvolution(image, psf, balance=0.01, eps=1e-5):
    # Wiener decon in the frequency domain
    # turn imageto frequency space
    image_ft = np.fft.fft2(image)

    # pad PSF with zeros so convolution sizes match (needed to avoid artefacts)
    psf_padded = np.zeros_like(image)
    psf_shape = psf.shape

    # center PSF in the middle of the image center
    y0 = (image.shape[0] - psf_shape[0]) // 2
    x0 = (image.shape[1] - psf_shape[1]) // 2
    # x0 an y0 are the top left corner of the PSF insertion into the zero-Array (same size as image)
    psf_padded[y0:y0 + psf_shape[0], x0:x0 + psf_shape[1]] = psf

    # shift PSF so its center aligns with (0,0) frequency after FFT
    psf_centered = np.fft.ifftshift(psf_padded)
    # turn PSF to frequency space
    psf_ft = np.fft.fft2(psf_centered)

    # Actual wiener filter with balance roughly being the snr Term
    psf_conj = np.conj(psf_ft)
    denom = psf_ft * psf_conj + balance
    # epsilon is addded for tiny safeguard incase the denom is too small at some points and noise
    # does not get too amplified
    deconvolved_ft = psf_conj * image_ft / (denom + eps)

    # turn back to spatial domain
    deconvolved = np.fft.ifft2(deconvolved_ft)
    return np.abs(deconvolved)
