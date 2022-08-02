import numpy as np
import seaborn as sns

sns.set()
from matplotlib import pyplot as plt

N = 50
Ns = [10, 30, 50, 100]
num_of_alleles = 2
num_of_experiment_repeat = 500
N_WITH_MUTATION = 1000
FREQ = [0.1, 1, 10]


def is_population_fixated(population) -> bool:
    return np.all(population == population[0])


def moran(population, N) -> int:
    """

    :param population:
    :return: time until fixation
    """
    num_iter = 0
    while not is_population_fixated(population):
        to_duplicate = np.random.randint(0, N)
        to_delete = np.random.randint(0, N)
        while to_delete == to_duplicate:
            to_delete = np.random.randint(0, N)
        population[to_delete] = population[to_duplicate]
        num_iter += 1
    return num_iter


def wright_fisher(population, N):
    _, counts = np.unique(population, return_counts=True)
    freq = counts / N
    num_iter = 0
    while 1 not in freq:
        freq = np.random.multinomial(N, freq) / N
        num_iter += 1
    return num_iter


def wright_fisher2(population, N, mutation_freq, num_generations):
    for i in range(num_generations):
        mutation_chosen = np.random.binomial(1, mutation_freq / N, N)
        amount_mutations = np.sum(mutation_chosen)
        new_mutations_values = np.random.randint(0, 2 ** 31, size=amount_mutations)
        population[mutation_chosen == 1] = new_mutations_values
    return len(np.unique(population))


# num_iters = np.zeros(num_of_experiment_repeat)
# for i in range(num_of_experiment_repeat):
#     population = np.ones(shape=N, dtype='uint8')
#     population[np.arange(N // 2)] = 0
#     num_iters[i] = moran(population)
#
# plt.title(f'frequencies of fixated moran model num of generations avg is {np.mean(num_iters).round(3)}')
# plt.hist(num_iters, bins=30)
# plt.show()

def simulation_q_1_2(N, is_moran=False, num_of_experiment_repeat=num_of_experiment_repeat, p=0.5, is_q4=False):
    if not is_q4:
        algo = wright_fisher
        name = "wright_fisher"
        if is_moran:
            algo = moran
            name = 'moran'
    else:
        algo = wright_fisher_q4
        name = "wright_fisher"
        if is_moran:
            algo = moran_q4
            name = 'moran'

    num_iters = np.zeros(num_of_experiment_repeat)
    for i in range(num_of_experiment_repeat):
        population = create_population(N, p)
        num_iters[i] = algo(population, N)

    plt.title(f'N={N} frequencies of fixated {name} model num of generations avg is {np.mean(num_iters).round(3)}')
    plt.hist(num_iters, bins=30)
    plt.show()


#
# for N in Ns:
#     simulation(N)
# for N in Ns:
#     simulation(N, True)

def create_population(N, p=0.5):
    population = np.ones(shape=N, dtype='uint8')
    population[np.arange(int(N * p))] = 0
    return population


def simulation_q_3(mutation_freq, to_print=True):
    num_generations = 300
    population = create_population(N_WITH_MUTATION)
    amount_of_different_alleles = wright_fisher2(population, N_WITH_MUTATION, mutation_freq, num_generations)
    if to_print:
        print(
            f'amount of different alleles for mutation rate={(mutation_freq / N).__round__(3)}'
            f' is {amount_of_different_alleles} for {num_generations} num of generations')
    return amount_of_different_alleles


for mutation_freq in FREQ:
    simulation_q_3(mutation_freq)

num_iters = 1000
for mutation_freq in FREQ:
    unique_alleles_array = np.zeros(num_iters)
    for i in range(num_iters):
        unique_alleles_array[i] = simulation_q_3(mutation_freq,to_print=False)
    plt.title(f'amount of different alleles for mutation rate={(mutation_freq/N_WITH_MUTATION).__round__(6)} distribution, avg:{np.mean(unique_alleles_array)}')
    plt.hist(unique_alleles_array, bins=12)
    plt.show()


def wright_fisher_q4(population, N):
    _, counts = np.unique(population, return_counts=True)
    freq = counts / N
    num_iter = 0
    fitness_dist = np.array([1.1 / 2.1, 1 / 2.1])
    while 1 not in freq:
        prob = (freq * fitness_dist)
        prob = prob/sum(prob)
        freq = np.random.multinomial(N, prob) / N
        num_iter += 1
    return num_iter


def moran_q4(population, N) -> int:
    """

    :param population:
    :return: time until fixation
    """
    num_iter = 0
    fitness_dist = np.array([1.1 / 2.1, 1 / 2.1])
    a = np.arange(N)
    while not is_population_fixated(population):
        _, counts = np.unique(population, return_counts=True)
        freq = counts / N
        prob = (freq * fitness_dist)
        prob = prob / sum(prob)
        type_to_dup = np.random.binomial(1, prob[1])
        indexes = a[population == type_to_dup]
        if not indexes.size:  # to fix for more than two alleles
            return num_iter
        to_duplicate = indexes[0]
        to_delete = np.random.randint(0, N)
        while to_delete == to_duplicate:
            to_delete = np.random.randint(0, N)
        population[to_delete] = population[to_duplicate]
        num_iter += 1
    return num_iter


Ns_q_4 = [50, 500, 10000]
# Ns_q_4 = [10000]
num_repeats = 500

# for N in Ns_q_4:
#     simulation_q_1_2(N, False, num_repeats, p=0.02, is_q4=True)

for N in Ns_q_4:
    simulation_q_1_2(N, True, num_repeats, p=0.02, is_q4=True)