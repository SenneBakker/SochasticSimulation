import random

from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import multiprocessing
import parameters as par
fig = plt.figure(figsize=(8,6))

# code from https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html , even checken of we dit wel mogen gebruiken

# Set seed for reproducibility
#np.random.seed(420)

#test
print('test git')


def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n


def monte_carlo():
    sample_area_count = 0
    for i in range(par.NO_SAMPLES):

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(par.RE_START + x * (par.RE_END - par.RE_START),
                    par.IM_START + y * (par.IM_END - par.IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, par.MAX_ITER)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / par.MAX_ITER)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def improved_monte_carlo(): # https://www.researchgate.net/publication/228755472_Strategies_for_Improving_the_Efficiency_of_Monte-Carlo_Methods
    sample_area_count = 0
    for i in range(par.NO_SAMPLES):

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(par.RE_START + x * (par.RE_END - par.RE_START),
                    par.IM_START + y * (par.IM_END - par.IM_START))
        # Compute the number of iterations
        m = mandelbrot(c, par.MAX_ITER)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / par.MAX_ITER)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def orthogonal_sampling():
    major = par.NO_SUBSPACES
    samples = major*major
    sample_area_count = 0
    m = 0
    xlist = np.empty((major, major))
    ylist = np.empty((major, major))
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / samples
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / samples
    print(scale_x, scale_y)

    for i in tqdm(range(0, major)):
        for j in range(0, major):
            xlist[i][j] = ylist[i][j] = m
            m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in tqdm(range(0, major)):
        for j in range(0, major):
            x = par.RE_START + (scale_x * (xlist[i][j]+ np.random.uniform(0,1)))
            y = par.IM_START + (scale_y * (ylist[j][i]+ np.random.uniform(0,1)))

            c = complex(x, y)
            # Compute the number of iterations
            m = mandelbrot(c, par.MAX_ITER)
            # The color depends on the number of iterations
            color = 255 - int(m * 255 / par.MAX_ITER)
            if color == 0:
                sample_area_count += 1
                plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
            else:
                plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)
            #plt.axvline(x=(par.RE_START + (scale_x * (xlist[i][j]))), color='black')
            #plt.axhline(y= (par.IM_START + (scale_y * (ylist[j][i]))), color='black')

    plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def latin_hypercube():
    sample_area_count = 0
    m = 0
    xlist = np.zeros(par.NO_SAMPLES)
    ylist = np.zeros(par.NO_SAMPLES)
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in tqdm(range(0, par.NO_SAMPLES)):
        xlist[i] = ylist[i] = m
        m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in tqdm(xlist):
        x = par.RE_START + scale_x * (xlist[int(i)] + np.random.uniform(0, 1))
        y_index = int(np.random.randint(0, len(ylist)))
        y = par.IM_START + scale_y * (ylist[y_index] + np.random.uniform(0, 1))
        #plt.axvline(x=(par.RE_START + (scale_x * (xlist[int(i)]))), color='black')
        #plt.axhline(y=(par.IM_START + (scale_y * (ylist[y_index]))), color='black')
        ylist = np.delete(ylist, y_index)
        c = complex(x, y)
        # Compute the number of iterations
        m = mandelbrot(c, par.MAX_ITER)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / par.MAX_ITER)
        if color == 0:
            sample_area_count += 1
            plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
        else:
            plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)

    plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def pure_random_sampling():
    sample_area_count = 0
    for i in range(par.NO_SAMPLES):

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=par.RE_START, high=par.RE_END)
        y = np.random.uniform(low=par.IM_START, high=par.IM_END)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(x,y)
        # Compute the number of iterations
        m = mandelbrot(c, par.MAX_ITER)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / par.MAX_ITER)
        if color == 0:
            sample_area_count += 1

    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def balanced_is(process_range):
    areas_js = []
    areas_diff = []
    area_is = monte_carlo()
    for j in tqdm(process_range): # van 1 tot 200 max iteraties
        area_js = monte_carlo(j)
        areas_js.append(area_js)
        area_diff = area_js - area_is
        areas_diff.append(area_diff)
    return areas_js, areas_diff


def balanced_sample_size(sizes):
    areas_js = []
    for j in tqdm(sizes): # van 1 tot 200 max iteraties
        par.NO_SAMPLES = j
        area_js = monte_carlo()
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

#print(latin_hypercube())
#print(orthogonal_sampling())
#print(pure_random_sampling())

'''
if __name__ == '__main__':
    pool = multiprocessing.Pool(os.cpu_count())
    iterations = list(range(1, par.MAX_ITER+1, 1))
    areas_js, areas_diff = pool.apply(balanced_is, (iterations,))
    plot_area_convergence(areas_js, areas_diff, iterations)
    sample_sizes = [10**j for j in range(2, 7)]
    sample_size_area = pool.apply(balanced_sample_size, (sample_sizes,))
    plot_sample_sizes(sample_sizes, sample_size_area)'''




