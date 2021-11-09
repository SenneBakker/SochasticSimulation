from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy.stats import qmc
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing
import parameters as par
fig = plt.figure(figsize=(8,6))

# code from https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html , even checken of we dit wel mogen gebruiken

# Set seed for reproducibility
np.random.seed(420)


def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n


def monte_carlo(max_iter):
    sample_area_count = 0
    for i in range(par.NO_SAMPLES): # tqdm used to plot progress bar

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(par.RE_START + x * (par.RE_END - par.RE_START),
                    par.IM_START + y * (par.IM_END - par.IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def latin_hypercube(n):
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n)
    l_bounds = [par.IM_START, par.RE_START]
    u_bounds = [par.IM_END, par.RE_END]
    samples = qmc.scale(sample, l_bounds, u_bounds)
    sample_area_count = 0
    for i in samples:
        x = i[0]
        y = i[1]

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(par.RE_START + x * (par.RE_END - par.RE_START),
                    par.IM_START + y * (par.IM_END - par.IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def balanced_is(process_range):
    areas_js = []
    areas_diff = []
    area_is = monte_carlo(par.MAX_ITER)
    for j in tqdm(process_range): # van 1 tot 200 max iteraties
        area_js = monte_carlo(j)
        areas_js.append(area_js)
        area_diff = area_js - area_is
        areas_diff.append(area_diff)
    return areas_js, areas_diff


def balanced_sample_size(sizes):
    areas_js = []
    areas_diff = []
    for j in tqdm(sizes): # van 1 tot 200 max iteraties
        par.NO_SAMPLES = j
        area_js = monte_carlo(par.MAX_ITER)
        areas_js.append(area_js)
    return areas_js


def plot_area_convergence(a_js, a_diff, its):
    plt.plot(its, a_js, color="r", label="area")
    plt.xlabel('Max iterations Mandelbrot')
    plt.ylabel('Area approximation')
    plt.savefig('A_js_%s_%s.jpg'%(par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    plt.plot(its, a_diff, color="g", label="diff_area")
    plt.xlabel('Max iterations Mandelbrot')
    plt.ylabel('A_js - A_is')
    plt.savefig('A_diff_%s_%s.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    return


def plot_sample_sizes(ss, ssz):
    plt.plot(ss, ssz, c='red')
    plt.xlabel('Sample size')
    plt.ylabel('Approximated area')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.savefig('Diff_samplesizes_area_%s.jpg' % (par.MAX_ITER))
    return


print(latin_hypercube(100))
'''

if __name__ == '__main__':
    pool = multiprocessing.Pool(os.cpu_count())
    iterations = list(range(1, par.MAX_ITER+1, 1))
    areas_js, areas_diff = pool.apply(balanced_is, (iterations,))
    plot_area_convergence(areas_js, areas_diff, iterations)
    sample_sizes = [10**j for j in range(2, 7)]
    sample_size_area = pool.apply(balanced_sample_size, (sample_sizes,))
    plot_sample_sizes(sample_sizes, sample_size_area) '''


