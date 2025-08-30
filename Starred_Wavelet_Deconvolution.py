import numpy as np
from starred.deconvolution.deconvolution import setup_model
from starred.deconvolution.parameters import ParametersDeconv
from starred.deconvolution.loss import Loss
from starred.optim.optimization import Optimizer
from starred.utils.noise_utils import propagate_noise

# Starred deconvolution algorithm implemented as instructed in the GITLAB page by COSMOGRAIL - Works but not utilized in final comparison
def starred_deconvolution(images, noisemaps, psfs, x_pos, y_pos, fluxes):
    model, k_init, k_up, k_down, k_fixed = setup_model(
        data=images,
        sigma_2=noisemaps**2,
        s=psfs,
        xs=x_pos, ys=y_pos,
        initial_a=fluxes,
        subsampling_factor=2
    )

    h0 = k_init['kwargs_background'].get('h', 1.0)
    k_init['kwargs_background']['h'] = 0.3 * h0
    k_fixed['kwargs_background'].pop('h', None)

    params = ParametersDeconv(k_init, k_fixed, k_up, k_down)

    eps = 1e-8
    wmap = 1.0 / np.maximum(noisemaps, eps)
    wmap = wmap / wmap.mean()

    # small change, here the first pass over the model has been changed to relax the deconvolution in an attempt to not deconvolve too much
    W0 = propagate_noise(model, noisemaps, k_init, upsampling_factor=2)[0]
    try:
        loss = Loss(images, model, params, noisemaps**2, W=W0, weight_map=wmap)
    except TypeError:
        loss = Loss(images, model, params, noisemaps**2, W=W0)

    Optimizer(loss, params, method='l-bfgs-b').minimize(maxiter=70)
    k_optim = params.best_fit_values(as_kwargs=True)

    W = propagate_noise(model, noisemaps, k_optim, upsampling_factor=2)[0]
    params = ParametersDeconv(k_optim, k_fixed, k_up, k_down)
    try:
        loss = Loss(images, model, params, noisemaps**2, W=W, weight_map=wmap)
    except TypeError:
        loss = Loss(images, model, params, noisemaps**2, W=W)

    Optimizer(loss, params, method='l-bfgs-b').minimize(maxiter=30)
    return model.getDeconvolved(params.best_fit_values(as_kwargs=True))