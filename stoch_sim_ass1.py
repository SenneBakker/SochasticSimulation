import random
import pandas as pd
from tqdm import tqdm   # used for time management
import numpy as np
import math
import pickle
import statistics as st
from scipy import stats
import os
import matplotlib.pyplot as plt
import multiprocessing   # used to improve time efficiency
import parameters as par
fig = plt.figure(figsize=(6,4), dpi=300)

# https://www.codingame.com/playgrounds/2358/how-to-plot-the-mandelbrot-set/mandelbrot-set
# https://www.researchgate.net/publication/228755472_Strategies_for_Improving_the_Efficiency_of_Monte-Carlo_Methods
# https://www.fractalus.com/kerry/articles/area/mandelbrot-area.html

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
    for i in range(par.NO_SAMPLES):

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


# improved monte carlo
def imp_monte_carlo(max_iter):
    sample_area_count = 0

    for idx, item in enumerate(range(par.NO_SAMPLES)):
        x = np.random.uniform(low=0, high=1)
        y = np.random.uniform(low=0, high=1)
        if idx % 2: # every even number we pick the antithetic variete of x or y
            x = 1 - x
            y = 1 - y
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


def orth_sampling(max_iter):
    major = int(math.sqrt(par.NO_SAMPLES))
    sample_area_count = 0
    m = 0
    xlist = np.empty((major, major))
    ylist = np.empty((major, major))
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, major):
        for j in range(0, major):
            xlist[i][j] = ylist[i][j] = m
            m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in range(0, major):
        for j in range(0, major):
            x = par.RE_START + (scale_x * (xlist[i][j]+ np.random.uniform(0,1)))
            y = par.IM_START + (scale_y * (ylist[j][i]+ np.random.uniform(0,1)))

            c = complex(x, y)
            # Compute the number of iterations
            m = mandelbrot(c, max_iter)
            # The color depends on the number of iterations
            color = 255 - int(m * 255 / max_iter)
            if color == 0:
                sample_area_count += 1
                #plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
            #else:
                #plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)
            #plt.axvline(x=(par.RE_START + (scale_x * (xlist[i][j]))), color='black')
            #plt.axhline(y= (par.IM_START + (scale_y * (ylist[j][i]))), color='black')

    #plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def lhs(max_iter):
    sample_area_count = 0
    m = 0
    xlist = np.zeros(par.NO_SAMPLES)
    ylist = np.zeros(par.NO_SAMPLES)
    scale_x = (abs(par.RE_START) + abs(par.RE_END)) / par.NO_SAMPLES
    scale_y = (abs(par.IM_START) + abs(par.IM_END)) / par.NO_SAMPLES

    for i in range(0, par.NO_SAMPLES):
        xlist[i] = ylist[i] = m
        m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)

    for i in xlist:
        x = par.RE_START + scale_x * (xlist[int(i)] + np.random.uniform(0, 1))
        y_index = int(np.random.randint(0, len(ylist)))
        y = par.IM_START + scale_y * (ylist[y_index] + np.random.uniform(0, 1))
        #plt.axvline(x=(par.RE_START + (scale_x * (xlist[int(i)]))), color='black')
        #plt.axhline(y=(par.IM_START + (scale_y * (ylist[y_index]))), color='black')
        ylist = np.delete(ylist, y_index)
        c = complex(x, y)
        # Compute the number of iterations
        m = mandelbrot(c, max_iter)
        # The color depends on the number of iterations
        color = 255 - int(m * 255 / max_iter)
        if color == 0:
            sample_area_count += 1
            #plt.plot(x, y, 'ro', color='green', ms=72. / fig.dpi)
        #else:
            #plt.plot(x, y, 'ro', color='red', ms=72. / fig.dpi)

    #plt.show()
    # Calculate average of the samples and determine area approx
    within_mb_set = sample_area_count / par.NO_SAMPLES
    area_approx = within_mb_set * par.AREA_SQUARE
    return area_approx


def pure_random(max_iter):
    sample_area_count = 0
    for i in range(par.NO_SAMPLES):

        # Draw random numbers from uniform distribution
        x = np.random.uniform(low=par.RE_START, high=par.RE_END)
        y = np.random.uniform(low=par.IM_START, high=par.IM_END)

        # Convert coordinates to complex number and compute the mandelbrot iterations
        c = complex(x,y)
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

# Function used to calculate the differences A_is - A_js , indicating convergence
def cpu_spread(process_range, function):
    areas_js = []
    areas_diff = []
    area_is = function(par.MAX_ITER)
    for j in tqdm(process_range): # van 1 tot 200 max iteraties
        area_js = function(j)
        areas_js.append(area_js)
        area_diff = area_js - area_is
        areas_diff.append(area_diff)
    return areas_js, areas_diff


# Function used to calculate the statistics for every method, iterations mandelbrot kept fixed
def cpu_spread_taking_maxiter(process_range, function):
    areas = []
    for j in tqdm(process_range):
        area = function(par.MAX_ITER)
        areas.append(area)

    avg_area = st.mean(areas)
    var_area = st.variance(areas)

    return avg_area, var_area, areas


def balanced_sample_size(sizes):
    areas_js = []
    for j in tqdm(sizes): # van 1 tot 200 max samples
        par.NO_SAMPLES = j
        area_js = monte_carlo()
        areas_js.append(area_js)
    return areas_js


def plot_area_convergence(result_dict):
    results = ["monte_carlo","pure_random","orth_sampling","lhs", "imp_monte_carlo"]
    # plot total area
    for i in results:
        plt.plot(result_dict["iterations"], result_dict[i]["area_js"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('Area approximation', fontsize=12)
    plt.legend()
    plt.savefig('A_js_%s_%s.jpg'%(par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    # plot diff area
    for i in results:
        plt.plot(result_dict["iterations"], result_dict[i]["area_diff"], label="%s"%i)
    plt.xlabel('Max iterations Mandelbrot', fontsize=12)
    plt.ylabel('A_js - A_is', fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig('A_diff_%s_%s.jpg' % (par.NO_SAMPLES, par.MAX_ITER))
    plt.clf()
    return


def perform_stat_analysis(stats_dict):
    # https://www.reneshbedre.com/blog/anova.html

    levenes_stat, pvalue2 = stats.levene(stats_dict["monte_carlo"]["areas"],
                                    stats_dict["pure_random"]["areas"],
                                    stats_dict["orth_sampling"]["areas"],
                                    stats_dict["lhs"]["areas"],
                                    stats_dict["imp_monte_carlo"]["areas"])
    print('we do not assume that the populations have equal variance, therefore levene instead of anova')
    print('levene digit: %s, pvalue: %s' %(levenes_stat, pvalue2))
    print('pvalue smaller than 0.05, therefore the variances differ significantly\n')
    df = pd.DataFrame.from_dict(stats_dict)
    df.drop(df.tail(1).index, inplace=True)# Skip last row with all areas
    df.to_excel("output.xlsx")
    print(df)
    return


def plot_sample_sizes(ss, ssz):
    plt.plot(ss, ssz, c='red')
    plt.xlabel('Sample size')
    plt.ylabel('Approximated area')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.savefig('Diff_samplesizes_area_%s.jpg' % (par.MAX_ITER))
    return


# select here what to run of the script
run_ajs_convergence = False
run_stats_max_iter = False
run_ajs_conv_plotting = False
run_statistical_analysis = True # to be completed

run_sample_sizes = False

results = {
    "iterations" : [],
    "monte_carlo": [],
    "pure_random": [],
    "orth_sampling": [],
    "lhs" : [],
    "imp_monte_carlo": []}

stats_dict ={"monte_carlo": [],
    "pure_random": [],
    "orth_sampling": [],
    "lhs" : [],
    "imp_monte_carlo": [],
    "stat_sample_size": 0
}


if __name__ == '__main__':
    if run_ajs_convergence:
        methods = [monte_carlo, pure_random, orth_sampling, lhs, imp_monte_carlo]
        iterations = list(range(1, par.MAX_ITER + 1, 1))
        for method in methods:
            print(str(method))
            pool = multiprocessing.Pool(os.cpu_count())
            areas_js, areas_diff = pool.apply(cpu_spread, (iterations, method, ))
            label = str(method.__name__)
            results[label] = {"area_js": areas_js,
                              "area_diff": areas_diff}
        results["iterations"] = iterations

        # Save data to pickle file so we can plot without running the algo's again
        file_to_write = open("results.pickle", "wb")
        pickle.dump(results, file_to_write)
        file_to_write.close()

    if run_stats_max_iter:
        stat_sample_size = 100 # 100 sample runs
        methods = [monte_carlo, pure_random, orth_sampling, lhs, imp_monte_carlo]
        iterations = list(range(1, stat_sample_size + 1, 1))
        for method in methods:
            print(str(method))
            pool = multiprocessing.Pool(os.cpu_count())
            avg_area, var_area, areas = pool.apply(cpu_spread_taking_maxiter, (iterations, method,))
            label = str(method.__name__)
            stats_dict[label] = {"avg_area": avg_area,
                              "var_area": var_area,
                                 "areas": areas}
        stats_dict["stat_sample_size"] = stat_sample_size
        file_to_write = open("stats.pickle", "wb")
        pickle.dump(stats_dict, file_to_write)
        file_to_write.close()
        table = pd.DataFrame.from_dict(stats_dict)
        print(table)

    if run_ajs_conv_plotting:
        infile = open("results.pickle",'rb')
        new_dict_results = pickle.load(infile)
        plot_area_convergence(new_dict_results)

    if run_statistical_analysis:
        infile = open("stats.pickle", 'rb')
        new_dict_stats = pickle.load(infile)
        perform_stat_analysis(new_dict_stats)


    if run_sample_sizes:
        pool = multiprocessing.Pool(os.cpu_count())
        sample_sizes = [10**j for j in range(2, 7)]
        sample_size_area = pool.apply(balanced_sample_size, (sample_sizes,))
        plot_sample_sizes(sample_sizes, sample_size_area)



