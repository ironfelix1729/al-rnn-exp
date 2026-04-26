"""metrics.py - state-space and power-spectrum metrics from the AL-RNN paper.

Verbatim from the user's reference (Brenner et al., NeurIPS 2024 supplementary).

  * state_space_divergence_binning(x_gen, x_true, n_bins)
        = KL(p_true || p_gen) on a 3-D histogram of the state, with Laplace
          smoothing.  Bounding box from the TRUE trajectory.

  * power_spectrum_error(X_true, X_gen, smoothing)
        = mean over coordinates of the Hellinger distance between the
          smoothed, normalised power spectra of X_true and X_gen.
"""
import torch as tc
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def calc_histogram(x, n_bins, min_, max_):
    dim_x = x.shape[1]

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = tc.ones(coordinates.shape[0], device=coordinates.device)
    if 'cuda' == coordinates.device.type:
        tens = tc.cuda.sparse.FloatTensor
    else:
        tens = tc.sparse.FloatTensor
    return tens(coordinates.t(), indices, size=size_).to_dense()


def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf


def kullback_leibler_divergence(p1, p2):
    if p1 is None or p2 is None:
        kl = tc.tensor([float('nan')])
    else:
        kl = (p1 * tc.log(p1 / p2)).sum()
    return kl


def state_space_divergence_binning(x_gen, x_true, n_bins=30):
    min_, max_ = x_true.min(0).values, x_true.max(0).values
    hist_gen = calc_histogram(x_gen, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return kullback_leibler_divergence(p_true, p_gen).item()


def compute_and_smooth_power_spectrum(x, smoothing):
    x_ = (x - x.mean()) / x.std()
    fft_real = np.fft.rfft(x_)
    ps = np.abs(fft_real) ** 2 * 2 / len(x_)
    ps_smoothed = gaussian_filter1d(ps, smoothing)
    return ps_smoothed / ps_smoothed.sum()


def hellinger_distance(p, q):
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))


def power_spectrum_error(X, X_gen, smoothing=20):
    dists = []
    for i in range(X.shape[1]):
        ps = compute_and_smooth_power_spectrum(X[:, i], smoothing)
        ps_gen = compute_and_smooth_power_spectrum(X_gen[:, i], smoothing)
        dists.append(hellinger_distance(ps, ps_gen))
    return float(np.mean(dists))
