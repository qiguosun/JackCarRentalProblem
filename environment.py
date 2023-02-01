import json
from itertools import product
from poissonDistribution import Poisson
import numpy as np
with open("config.json") as config_file:
    data = json.load(config_file)
    MU = data["MU"]
    STATESPACE = data["STATESPACE"]
    EARN_PER_CAR = data["EARN_PER_CAR"]


class Environment(object):
    def __init__(self, EARN_PER_CAR=EARN_PER_CAR, STATESPACE=STATESPACE):
        self.earn_per_car = EARN_PER_CAR
        Poisson_Distribution = Poisson()
        self.possion_distribution = Poisson_Distribution.create_distribution()
        self.stateSpace = STATESPACE
        self.expect_profit = np.zeros([self.stateSpace[0], self.stateSpace[0]])
        self.prob = np.zeros(
            [self.stateSpace[0], self.stateSpace[0], self.stateSpace[0], self.stateSpace[0]])

    def calc_reward_and_prob(self):
        for cars_at_A, cars_at_B in product(list(range(self.stateSpace[0])), list(range(self.stateSpace[0]))):
            for A_rent_cars, B_rent_cars in product(list(range(cars_at_A+1)), list(range(cars_at_B+1))):
                profit = self.earn_per_car * (A_rent_cars+B_rent_cars)
                prob_A_rent_cars = self.possion_distribution[1, A_rent_cars, int(
                    A_rent_cars == cars_at_A)]
                prob_B_rent_cars = self.possion_distribution[2, B_rent_cars, int(
                    B_rent_cars == cars_at_B)]
                prob_rent_pair = prob_A_rent_cars * prob_B_rent_cars
                self.expect_profit[cars_at_A,
                                   cars_at_B] += profit * prob_rent_pair
                A_return_bound, B_return_bound = self.stateSpace[0] + \
                    A_rent_cars - \
                    cars_at_A, self.stateSpace[0] + B_rent_cars-cars_at_B
                for A_return_cars, B_return_cars in product(list(range(A_return_bound)), list(range(B_return_bound))):
                    prob_A_return_cars = self.possion_distribution[1, A_return_cars, int(
                        A_return_cars == A_return_bound-1)]
                    prob_B_return_cars = self.possion_distribution[0, B_return_cars, int(
                        B_return_cars == B_return_bound-1)]
                    prob_return_pair = prob_A_return_cars * prob_B_return_cars
                    final_cars_at_A = cars_at_A-A_rent_cars+A_return_cars
                    final_cars_at_B = cars_at_B-B_rent_cars+B_return_cars
                    self.prob[final_cars_at_A, final_cars_at_B, cars_at_A,
                              cars_at_B] += prob_rent_pair * prob_return_pair
        return self.expect_profit, self.prob


if __name__ == "__main__":
    environment = Environment()
    reward, transition = environment.calc_reward_and_prob()
    print(reward[3, 5])
