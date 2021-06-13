from scipy.optimize import linprog
import random
import time

num_agents = 30

def find_opt_lp(network_coefficients,distances,reservation_prices):
    obj = [-distances[i] for i in range(num_agents)] + [0 for i in range(num_agents)]

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

    rhs_eq = [reservation_prices[i] for i in range(len(lhs_eq))]
    
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd,
                  A_eq=lhs_eq, b_eq=rhs_eq,
                  method="simplex")

    return {'value':opt.fun*-1,'prices':list(opt.x[:num_agents])}

def find_opt_dp(network_coefficients,distances,reservation_prices):
    def find_sum(agent_num,l,network_coefficients):
        sum_paths = [0 for i in range(len(l))]
        for j in range(len(l)-2,agent_num-1,-1):
            if l[j] == "max" and j!=agent_num:
                sum_paths[j] = 0
            else:
                for k in range(j+1,len(l)):
                    if l[k]!='min':
                        sum_paths[j]+=distances[k]*network_coefficients[k][j]*(1+sum_paths[k])
                    else:
                        sum_paths[j]+=network_coefficients[k][j]*(sum_paths[k])



        return sum_paths[agent_num]
    l = ["max" for i in range(len(network_coefficients))]
    for i in range(len(network_coefficients)-1,-1,-1):
        c = find_sum(i,l,network_coefficients)
        if c>=distances[i]:
            l[i] = "min"

    prices = []
    utilities = []

    for i in range(len(network_coefficients)):
        sum_value = reservation_prices[i]
        for j in range(len(network_coefficients[i])):
            sum_value+=utilities[j]*network_coefficients[i][j]
        if l[i] == "min":
            prices.append(0)
            utilities.append(sum_value)
        else:
            prices.append(sum_value)
            utilities.append(0)
    
    return {'prices':prices,'value':sum([prices[i]*distances[i] for i in range(len(prices))])}

def get_utility_values(network_coefficients,prices,reservation_price=1):
    utilities = []

    for i in range(num_agents):
        utility = reservation_price-prices[i]

        for j in range(len(network_coefficients[i])):
            utility+=network_coefficients[i][j]*utilities[j]
        utilities.append(utility)


    utilities = [round(i,2) for i in utilities]

    return utilities
    

if __name__ == '__main__':
    start = time.time()

    for i in range(100):
        network_coefficients = []
        #network_coefficients = [[], [0.8031030457446929], [0.03228930727042534, 0.7154714402389166]]
        if len(network_coefficients) == 0:    
            for i in range(num_agents):
                temp = []
                for j in range(i):
                    temp.append(random.random())
                network_coefficients.append(temp)

        distances = [random.random() for i in range(num_agents)]
        #distances = [0.05725216897162699, 0.0346395191371186, 0.8137870490761298]

        reservation_prices = [random.random() for i in range(num_agents)]

        opt_value = find_opt_lp(network_coefficients,distances,reservation_prices)
        utilities = get_utility_values(network_coefficients,opt_value['prices'])
        predicted = find_opt_dp(network_coefficients,distances,reservation_prices)

        if abs(predicted['value']-opt_value['value'])>.01:
            print(opt_value['prices'],predicted['prices'])
            print(opt_value['value'],predicted['value'])
            print(network_coefficients)
            print(distances)
            break


    print("Took {} seconds".format(time.time()-start))
