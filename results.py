import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np
from itertools import product
import json
import pandas as pd
import seaborn as sns
from policyIteration import PolicyIteration
with open("config.json") as config_file:
    data = json.load(config_file)
    STATESPACE = data["STATESPACE"]

solver = PolicyIteration()
opt_policy, opt_value = solver.policy_iteration_solver()

df_policy = pd.DataFrame(
    opt_policy[::-1],
    index=list(range(STATESPACE[0]-1, -1, -1)),
    columns=list(range(STATESPACE[0]))
)
plt.figure(figsize=(10, 8))
sns.heatmap(data=df_policy, vmin=-5, vmax=5, square=True, cmap="Blues_r")
plt.title('Policy (Jack\'s Car Rental)')
plt.xlabel('Cars at B')
plt.ylabel('Cars at A')
plt.savefig("./fig/policy.png")
