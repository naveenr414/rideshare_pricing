from scipy.optimize import linprog
import random
import time

num_agents = 20

def find_opt_lp(network_coefficients,reservation_price=1):
    obj = [-1 for i in range(num_agents)] + [0 for i in range(num_agents)]

    lhs_ineq = []
    for i in range(num_agents):
        temp = [0 for j in range(num_agents*2)]
        temp[i+num_agents] = -1
        lhs_ineq.append(temp)
    
    rhs_ineq = [0 for i in range(num_agents)]
    bnd=[(0,float("inf")) for i in range(num_agents*2)]

    lhs_eq = []
    for i in range(num_agents):
        temp = [0 for i in range(num_agents*2)]
        temp[i] = 1
        temp[i+num_agents] = 1
        for j in range(num_agents,i+num_agents):
            temp[j] = -network_coefficients[i][j-num_agents]
        lhs_eq.append(temp)

    rhs_eq = [reservation_price for i in range(len(lhs_eq))]
    
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd,
                  A_eq=lhs_eq, b_eq=rhs_eq,
                  method="simplex")

    return {'value':opt.fun*-1,'prices':opt.x[:num_agents]}

def find_opt_dp(network_coefficients):
    def find_sum(agent_num,l,network_coefficients):
        sum_paths = [0 for i in range(len(l))]
        for j in range(len(l)-2,agent_num-1,-1):
            if l[j] == "max" and j!=agent_num:
                sum_paths[j] = 0
            else:
                for k in range(j+1,len(l)):
                    if l[k]!='min':
                        sum_paths[j]+=network_coefficients[k][j]*(1+sum_paths[k])
                    else:
                        sum_paths[j]+=network_coefficients[k][j]*(sum_paths[k])
        return sum_paths[agent_num]
    l = ["max" for i in range(len(network_coefficients))]
    for i in range(len(network_coefficients)-1,-1,-1):
        c = find_sum(i,l,network_coefficients)
        if c>=1:
            l[i] = "min"
    return l

def get_utility_values(network_coefficients,prices,reservation_price=1):
    utilities = []

    for i in range(num_agents):
        utility = reservation_price-prices[i]

        for j in range(len(network_coefficients[i])):
            utility+=network_coefficients[i][j]*utilities[j]
        utilities.append(utility)


    utilities = [round(i,2) for i in utilities]

    return utilities
    

start = time.time()

for i in range(100):
    network_coefficients = []
    if len(network_coefficients) == 0:    
        for i in range(num_agents):
            temp = []
            for j in range(i):
                temp.append(random.random())
            network_coefficients.append(temp)

    opt_value = find_opt_lp(network_coefficients)
    utilities = get_utility_values(network_coefficients,opt_value['prices'])

    actual = [j == 0 for j in list(opt_value['prices'])]
    actual = [['max','min'][int(j)] for j in actual]

    predicted = find_opt_dp(network_coefficients)

    if actual!=predicted:
        print(network_coefficients)


print("Took {} seconds".format(time.time()-start))
