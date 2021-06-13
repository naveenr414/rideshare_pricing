from lp_solver import *
import matplotlib.pyplot as plt

num_agents = 10
network_coefficients = []
for i in range(num_agents):
    temp = []
    for j in range(i):
        temp.append(random.random())
    network_coefficients.append(temp)

reservation_prices = [random.random() for i in range(num_agents)]

distances = [1 for i in range(num_agents)]

xticks = []
values = []

for i in range(100):
    network_coefficients[-1][-1] = i/10

    xticks.append(i/10)
    values.append(find_opt_dp(network_coefficients,distances,reservation_prices)['value'])

plt.plot(xticks,values)
plt.xlabel("Distance values")
plt.ylabel("Total profit")
plt.title("Profit vs. distance")
plt.tight_layout()
plt.show()
