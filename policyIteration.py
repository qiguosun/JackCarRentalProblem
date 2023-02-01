import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np
from itertools import product
import json
from environment import Environment
with open("config.json") as config_file:
    data = json.load(config_file)
    MU = data["MU"]
    STATESPACE = data["STATESPACE"]
    EPS = data["Eps"]
    GAMMA = data["GAMMA"]
    MaxMovingCars = data["MaxMoving"]


class PolicyIteration(object):
    def __init__(self, EPS=EPS, STATESPACE=STATESPACE, GAMMA=GAMMA, MaxMovingCars=MaxMovingCars):
        self.eps = EPS
        self.stateSpace = STATESPACE
        self.GAMMA = GAMMA
        self.maxMovingCars = MaxMovingCars
        CarsRentEnviroment = Environment()
        self.reward, self.transition_prob = CarsRentEnviroment.calc_reward_and_prob()

    def get_value(self, cars_at_A, cars_at_B, action):
        cars_at_A = min(self.stateSpace[0]-1, int(cars_at_A+action))
        cars_at_B = min(self.stateSpace[0]-1, int(cars_at_B-action))
        profit = self.reward[cars_at_A, cars_at_B] + self.GAMMA * (
            self.value*self.transition_prob[:, :, cars_at_A, cars_at_B]).sum()-2*abs(action)
        return profit

    def policy_evaluation(self):
        delta = float('inf')
        while delta > self.eps:
            delta = 0
            for cars_at_A, cars_at_B in product(list(range(self.stateSpace[0])), list(range(self.stateSpace[0]))):
                old_profit = self.value[cars_at_A, cars_at_B]
                action = self.policy[cars_at_A, cars_at_B]
                new_profit = self.get_value(cars_at_A, cars_at_B, action)
                self.value[cars_at_A, cars_at_B] = new_profit

                delta = max(delta, abs(old_profit - new_profit))
        return

    def policy_improve(self):
        policy_stable = True
        for cars_at_A, cars_at_B in product(list(range(self.stateSpace[0])), list(range(self.stateSpace[0]))):
            action = self.policy[cars_at_A, cars_at_B]

            actions_valid = [action for action in range(
                max(-cars_at_A, -self.maxMovingCars), min(cars_at_B, self.maxMovingCars)+1)]
            values_valid = [self.get_value(
                cars_at_A, cars_at_B, action) for action in actions_valid]
            new_action = actions_valid[np.argmax(values_valid)]
            self.policy[cars_at_A, cars_at_B] = new_action

            if action != new_action:
                policy_stable = False
        return policy_stable

    def policy_iteration_solver(self):
        self.value = np.zeros([self.stateSpace[0], self.stateSpace[0]])
        self.policy = np.zeros([self.stateSpace[0], self.stateSpace[0]])
        policy_stable = False
        while not policy_stable:
            self.policy_evaluation()
            policy_stable = self.policy_improve()
        return self.policy, self.value


if __name__ == "__main__":
    solver = PolicyIteration()
    opt_policy, opt_value = solver.policy_iteration_solver()
    print(opt_policy[1, 1])
