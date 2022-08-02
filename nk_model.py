from typing import Dict, Tuple, Set

import numpy as np
from itertools import product
from matplotlib import pyplot as plt
import operator
import statsmodels.api as sm


N = 14
ALPHABET = {0: 0, 1: 1}
Ks = {3, 4, 10}


def autocorr(x, length=10):
    return sm.tsa.acf(x, nlags=length-1)



def autocorr5(x, lags):
    '''numpy.correlate, non partial'''
    mean = x.mean()
    var = np.var(x)
    xp = x - mean
    corr = np.correlate(xp, xp, 'full')[len(x) - 1:] / var / len(x)

    return corr[:len(lags)]


def autocorr2(data):
    size = 2 ** np.ceil(np.log2(2 * len(data) - 1)).astype('int')

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - np.mean(data)

    # Compute the FFT
    fft = np.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = np.fft.ifft(pwr).real / var / len(data)
    return acorr


def gen_group(k: int):
    '''
    return a set of all k+1 choose alphabet
    '''
    return set(product(ALPHABET.keys(), repeat=k + 1))


def gen_fitness_map(key: Tuple[int]) -> Dict[Tuple[int], float]:
    '''return a local map for all possible k sizes '''
    return {key: np.random.uniform(0, 1)}


def local_fitness_values_dict(N: int, k: int) -> Dict[int, Dict[Tuple[int], float]]:
    '''
    create a dictionary of dictionaries where first key is index and the value is dictionary
    (key is a tuple of : (index of combination, the vector) and the value is the fitness)
    :param N:
    :param k:
    :return:
    '''

    keys = gen_group(k)
    fitness_dict = dict()
    for index in range(N):
        dict_ind = dict()
        for key in keys:
            dict_ind.update(gen_fitness_map(key))
        fitness_dict[index] = dict_ind
    return fitness_dict


def genotype_fitness(genotype, N, k, local_fitness_values_dict):
    """ return the fitness of vector (duplicate the vector to calculate fitness for k > 0) """
    genotype2 = np.concatenate((genotype, genotype), axis=None)
    fitness = 0
    for index in range(N):
        key_tuple = tuple(genotype2[index:index + k + 1])
        fitness += local_fitness_values_dict[index][key_tuple]
    return fitness / N  # normalized by len of the vector


def get_random_genotype_fitness_over_trajectory(trajectory_length, N, k, local_fitness_values_dict):
    '''
    generate a random genotype sized N and calculate the fitness over a trajectory of length trajectory_length
    '''
    genotype0 = get_random_genotype(N)

    genotype_to_fitness = dict()
    genotype_random_walk_fitness = np.zeros(shape=(trajectory_length))

    genotype0_key = tuple(genotype0)
    genotype_to_fitness[genotype0_key] = genotype_fitness(genotype0, N, k, local_fitness_values_dict)
    genotype_random_walk_fitness[0] = genotype_to_fitness[genotype0_key]

    last_genotype = genotype0

    for i in range(1, trajectory_length):
        index, letter = get_index_and_letter_to_replace(N, last_genotype)

        last_genotype[index] = letter
        last_genotype_key = tuple(last_genotype)

        if last_genotype_key not in genotype_to_fitness.keys():
            cur_fitness = genotype_fitness(last_genotype, N, k, local_fitness_values_dict)
            genotype_to_fitness[last_genotype_key] = cur_fitness
        genotype_random_walk_fitness[i] = genotype_to_fitness[last_genotype_key]

    return genotype_random_walk_fitness


def get_index_and_letter_to_replace(N, last_genotype):
    index = np.random.randint(0, N)
    letter = np.random.choice(list(ALPHABET.keys()))
    while letter == last_genotype[index]:
        letter = np.random.choice(list(ALPHABET.keys()))
    return index, letter


def get_random_genotype(N):
    return np.random.choice(list(ALPHABET.keys()), N)


def count_landscape_local_max(N, k, local_fitness_values_dict):
    '''get as parameters the size of the genotype N the mutuaility constant k and local_fitness_values dictionary and
    return the amount of local maximas'''
    genotype_to_fitness = dict()
    count = 0
    for genotype in product(ALPHABET.keys(), repeat=N):
        count += is_local_maxima(np.array(genotype), local_fitness_values_dict, k, genotype_to_fitness)
    return count


def is_local_maxima(genotype, local_fitness_values_dict, k, genotype_to_fitness):
    key_genotype = tuple(genotype)
    if not genotype_to_fitness.get(key_genotype):
        genotype_to_fitness[key_genotype] = genotype_fitness(genotype=genotype, N=genotype.size, k=k,
                                                             local_fitness_values_dict=local_fitness_values_dict)

    original_fitness_of_genotype = genotype_to_fitness[key_genotype]

    for i in range(len(genotype)):
        for key in ALPHABET.keys():
            cur_gen = genotype.copy()
            cur_gen[i] = ALPHABET[key]
            key_cur_gen = tuple(cur_gen)
            if not genotype_to_fitness.get(key_cur_gen):
                genotype_to_fitness[key_cur_gen] = genotype_fitness(genotype=cur_gen, N=cur_gen.size, k=k,
                                                                    local_fitness_values_dict=local_fitness_values_dict)
            cur_fitness = genotype_to_fitness.get(key_cur_gen)
            if cur_fitness > original_fitness_of_genotype:
                return False
    return True


def count_genotype_trajectory_length(genotype, k, local_fitness_values_dict):
    genotype_to_fitness = dict()
    trajectory_length = 0
    while not is_local_maxima(genotype=genotype, local_fitness_values_dict=local_fitness_values_dict, k=k,
                              genotype_to_fitness=genotype_to_fitness):
        trajectory_length += 1
        genotype = np.array(get_highest_fitness_genotype_neighbour(genotype=genotype, k=k,
                                                                   genotype_to_fitness=genotype_to_fitness,
                                                                   local_fitness_values_dict=local_fitness_values_dict))
    return trajectory_length


def argmax(iterable):
    return max(iterable.items(), key=operator.itemgetter(1))[0]


def get_highest_fitness_genotype_neighbour(genotype, k, genotype_to_fitness, local_fitness_values_dict):
    neighbour_to_fitness = dict()
    for i in range(genotype.size):
        for letter in ALPHABET.keys():
            if letter == genotype[i]:
                continue

            neighbour_genotype = genotype.copy()
            neighbour_genotype[i] = letter
            neighbour_genotype_key = tuple(neighbour_genotype)
            if not genotype_to_fitness.get(neighbour_genotype_key):
                genotype_to_fitness[neighbour_genotype_key] = genotype_fitness(genotype=neighbour_genotype,
                                                                               N=neighbour_genotype.size, k=k,
                                                                               local_fitness_values_dict=
                                                                               local_fitness_values_dict)
            neighbour_to_fitness[neighbour_genotype_key] = genotype_to_fitness[neighbour_genotype_key]
    ret = argmax(neighbour_to_fitness)
    return ret


# def test1():


#     for k in Ks:
#         local_fitness_values = local_fitness_values_dict(N, k)
#         genotype = np.random.randint(low=0, high=2,size=N)
#         print(genotype_fitness(genotype, N, k, local_fitness_values))
#
#
# test1()


def test2():
    N = 7
    k = 4
    local_fitness_values = local_fitness_values_dict(N, k)
    autocorr_res = autocorr(get_random_genotype_fitness_over_trajectory(2 ** 5, N, k, local_fitness_values))
    plt.plot(autocorr_res)
    plt.show()


# test2()
#
# len_road = 10
# result = np.zeros((len(Ks), len_road))
# num_loops = 30
# for i in range(num_loops):
#     for j, k in enumerate(Ks):
#         local_fitness_values = local_fitness_values_dict(N=N, k=k)
#         vec = get_random_vec(num_genotypes=len_road, N=N, k=k,
#                              local_fitness_values_dict=local_fitness_values)[0]
#         autocorr_res = autocorr5(vec, lags=vec)
#         result[j] += autocorr_res
# result = result / num_loops
#
# for j, k in enumerate(Ks):
#     plt.plot(result[j], label=f"k={k}")
# plt.legend()
# plt.show()
#

#
# def local_fitness_values_dict2(N: int, k: int) -> Dict[int, Dict[Tuple[int], float]]:
#     '''
#     create a diciionary of dictionary where first key is index and the value is the values
#      is dictionary(key is a tuple of an alphabet combinations and the value is the fitness)
#     :param N:
#     :param k:
#     :return:
#     '''
#
#     keys = gen_group(k)
#     dict_ind = dict()
#     for key in keys:
#         dict_ind.update(gen_fitness_map(key))
#     return dict_ind
#
#
#
# def genotype_fitness2(genotype, N, k, local_fitness_values_dict):
#     genotype2 = np.concatenate((genotype, genotype), axis=None)
#     fitness = 0
#     for index in range(N):
#         key_tuple = tuple(genotype2[index:index + k + 1])
#         fitness += local_fitness_values_dict[key_tuple]
#     return fitness / N
#
#
# def get_random_vec2(num_genotypes, N, k, local_fitness_values_dict):
#     genotype0 = np.random.choice(list(ALPHABET.keys()), N)
#     genotype_to_fitness = dict()
#     genotype_random_walk_fitness = np.zeros(shape=(num_genotypes))
#     genotype_to_fitness[tuple(genotype0)] = genotype_fitness2(genotype0, N, k, local_fitness_values_dict)
#     genotype_random_walk_fitness[0] = genotype_to_fitness[tuple(genotype0)]
#     last_genotype = genotype0
#     for i in range(1, num_genotypes):
#         index = np.random.randint(0, N)
#         letter = np.random.choice(list(ALPHABET.keys()))
#         while letter == last_genotype[index]:
#             letter = np.random.choice(list(ALPHABET.keys()))
#         last_genotype[index] = letter
#         last_genotype_key = tuple(last_genotype)
#         if last_genotype_key not in genotype_to_fitness.keys():
#             cur_fitness = genotype_fitness2(last_genotype, N, k, local_fitness_values_dict)
#             genotype_to_fitness[last_genotype_key] = cur_fitness
#             genotype_random_walk_fitness[i] = cur_fitness
#         else:
#             genotype_random_walk_fitness[i] = genotype_to_fitness[last_genotype_key]
#     return genotype_random_walk_fitness, genotype_to_fitness

#
# len_road = 10
# result = np.zeros((len(Ks), len_road))
# num_loops = 100
# for k in [3]:
#     local_fitness_values = local_fitness_values_dict(N=N, k=k)
#     vec = get_random_vec2(num_genotypes=len_road, N=N, k=k,
#                           local_fitness_values_dict=local_fitness_values)[0]
#     plt.hist(vec)
#     plt.show()
# for i in range(num_loops):
#     for j, k in enumerate(Ks):
#         local_fitness_values = local_fitness_values_dict2(N=N, k=k)
#         vec = get_random_vec2(num_genotypes=len_road, N=N, k=k,
#                               local_fitness_values_dict=local_fitness_values)[0]
#         plt.hist(vec)
#         plt.show()
#         autocorr_res = autocorr5(vec, lags=vec)
#         result[j] += autocorr_res
# result = result / num_loops
#
# for j, k in enumerate(Ks):
#     plt.plot(result[j], label=f"k={k}")
# plt.legend()
# plt.show()
def test_i():
    num_repeats = 100
    N = 14
    k = 0
    trajectory_length = 2 ** 5
    Ks = [0, 4, 13]
    for k in Ks:
        auto_corr_cum = np.zeros(trajectory_length)
        for i in range(num_repeats):
            local_fitness_values = local_fitness_values_dict(N, k)
            fitness_over_trajectory = get_random_genotype_fitness_over_trajectory(trajectory_length=trajectory_length,
                                                                                  N=N, k=k,
                                                                                  local_fitness_values_dict=local_fitness_values)
            auto_corr_cum += autocorr(fitness_over_trajectory, length=trajectory_length)
        plt.plot(auto_corr_cum / num_repeats, label=f'{k}')
    plt.legend()
    plt.title(f'autocorrlation over k of trajectoies of length {trajectory_length} for {num_repeats} repeats averaged')
    plt.xlabel('length trajectory')
    plt.ylabel('autocorrelation')
    plt.show()


def test_ii():
    num_repeats = 10
    maximas_count_avg = np.zeros(N)
    for k in range(N):
        count = 0
        for i in range(num_repeats):
            local_fitness_values = local_fitness_values_dict(N, k)
            count += count_landscape_local_max(N=N, k=k, local_fitness_values_dict=local_fitness_values)
        maximas_count_avg[k] = count / num_repeats
    plt.title(label=f'number of local maxima for each k between 0 to {N - 1} averaged on {num_repeats} trials')
    plt.plot(maximas_count_avg)
    plt.xlabel('k')
    plt.ylabel('average number of local maxima')
    plt.show()


def test_iii():
    num_repeats = 50
    N = 14
    trajectory_length_avg = np.zeros(N)
    for k in range(N):
        trajectory_length = 0
        for i in range(num_repeats):
            local_fitness_values = local_fitness_values_dict(N, k)
            trajectory_length += count_genotype_trajectory_length(genotype=get_random_genotype(N), k=k,
                                                                  local_fitness_values_dict=local_fitness_values)
        trajectory_length_avg[k] = trajectory_length / num_repeats
    plt.title(label=f'trajectory average length averaged on {num_repeats} trials')
    plt.plot(trajectory_length_avg)
    plt.xlabel('k')
    plt.ylabel('trajectory average length')
    plt.show()


if __name__ == "__main__":
    test_i()
    # test_ii()
    # test_iii()
    # hey