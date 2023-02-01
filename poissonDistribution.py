import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np
from itertools import product
import json

with open("config.json") as config_file:
    data = json.load(config_file)
    MU = data["MU"]
    STATESPACE = data["STATESPACE"]


class Poisson(object):
    def __init__(self, MU=MU, STATESPACE=STATESPACE):
        self.mu = MU
        self.stateSpace = STATESPACE
        self.distribution = np.zeros([len(MU), STATESPACE[0], STATESPACE[1]])

    def create_distribution(self):
        for i, m in enumerate(self.mu):
            for n in range(self.stateSpace[0]):
                self.distribution[i, n, 0] = poisson.pmf(n, m)
                self.distribution[i, n, 1] = poisson.sf(n-1, m)
        return self.distribution


if __name__ == "__main__":
    Poisson_generator = Poisson()
    poisson_distribution = Poisson_generator.create_distribution()
    print(poisson_distribution[0, 0, 0])
