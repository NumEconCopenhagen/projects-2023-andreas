
from types import SimpleNamespace

import numpy as np
from scipy import optimize
from scipy.optimize import differential_evolution, LinearConstraint, Bounds

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0.0:
            H = np.min(HM,HF)

        elif par.sigma == 1.0:
            H = HM**(1-par.alpha)*HF**par.alpha

        else:
            exponent = (par.sigma-1)/par.sigma

            # special case for negative exponent as we cannot divide by 0
            if exponent < 0.0:
                
                # initialize terms of utility
                term1 = np.full(HM.size,np.inf)
                term2 = np.full(HM.size,np.inf)
                H = np.zeros(HM.size)

                # only for non-zero indices of HM and HF
                zero_indices_HM = (HM == 0) 
                zero_indices_HF = (HF == 0)
                term1[~zero_indices_HM] = (1 - par.alpha) * HM[~zero_indices_HM]**exponent
                term2[~zero_indices_HF] = par.alpha * HF[~zero_indices_HF]**exponent
                
                # only for non-infinity indices of term1 and term2
                inf_indices = (term1 == np.inf) | (term2 == np.inf)
                H[~inf_indices] = (term1[~inf_indices] + term2[~inf_indices])**(1/exponent)
            
            else:
                term1 = (1 - par.alpha) * HM**exponent
                term2 = par.alpha * HF**exponent
                H = (term1 + term2)**(1/exponent)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def calc_utility_(self,x):
        """ calculate utility """

        # a. consumption of market goods
        C = self.par.wM*x[0] + self.par.wF*x[2]

        # b. home production
        if self.par.sigma == 0.0:
            H = np.min(x[1],x[3])

        elif self.par.sigma == 1.0:
            H = x[1]**(1-self.par.alpha)*x[3]**self.par.alpha

        else:
            exponent = (self.par.sigma-1)/self.par.sigma
            term1 = (1 - self.par.alpha) * x[1]**exponent
            term2 = self.par.alpha * x[3]**exponent
            H = (term1 + term2)**(1/exponent)

        # c. total consumption utility
        Q = C**self.par.omega*H**(1-self.par.omega)
        utility = np.fmax(Q,1e-8)**(1-self.par.rho)/(1-self.par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/self.par.epsilon
        TM = x[0]+x[1]
        TF = x[2]+x[3]
        disutility = self.par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return - utility + disutility    

    def solve_discrete(self,do_print=False):
        """ solve model discretely """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        # a. contraint function (negative if violated)
        # Define the constraint matrices and vectors
        lc = LinearConstraint([[1,1,0,0],[0,0,1,1]],[0,0],[24,24])

        # Create a list of LinearConstraint object

        # b. call optimizer
        res = optimize.differential_evolution(self.calc_utility_,bounds=[(0,24),(0,24),(0,24),(0,24)],constraints=lc)

        return res.x

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        # a. output of this function to be minimized
        def squared_dev(x):

            # set alpha and beta
            setattr(self.par,'sigma',x[0])
            setattr(self.par,'alpha',x[1])
            
            # true beta0 and beta1
            beta = np.array([0.4,-0.1])

            # list of wages to solve model for
            wages = [0.8,0.9,1.0,1.1,1.2]

            # solve the model for different wages
            self.solve_multi_par('wF',wages,discrete=False)

            # run regression and save output
            beta_hat = self.run_regression()
            self.sol.beta0 = beta_hat[0]
            self.sol.beta1 = beta_hat[1]

            print('Squared deviation: ' + str(np.sum((beta - beta_hat) ** 2)))

            return np.sum((beta - beta_hat) ** 2)

        # array = np.zeros(2)
        # array[0] = 1.0
        # list = np.arange(0.0,1.0,0.1).tolist()
        # list_ = []
        # sqd = []

        # for x in list:
        #     new_array = np.copy(array)  # create a new copy of the array
        #     new_array[1] = x  # modify the copy
        #     list_.append(new_array)  # append the copy to the list

        # for x in list_:
        #     sqd.append(squared_dev(x))

        # plt.scatter(list,sqd)
        # plt.show()

        # b. call optimizer

        # Create a list of LinearConstraint object

        # b. call optimizer
        res = optimize.differential_evolution(squared_dev,bounds=[(0,2),(0,1)],tol=3.0)

        # print(res.message) # check that the solver has terminated correctly

    def solve_multi_par(self,par_name,par_list,discrete):
        'solve model for multiple parameters'

        # a. store solutions in array
        self.sol.array = np.zeros((len(par_list),4))

        # b1. solve discretely
        if discrete == True:

            # loop over different values
            for i,value in enumerate(par_list):
                    
                setattr(self.par,par_name,value) # set parameter value
                opt = self.solve_discrete(do_print=False) # solve the model discretely
                self.sol.array[i,:] = np.array([opt.LM,opt.HM,opt.LF,opt.HF]) # save solutions
        
        # b2. solve continuously
        else:
            
            # loop over different values
            for i,value in enumerate(par_list):
                    
                setattr(self.par,par_name,value) # set parameter value
                opt = self.solve(do_print=False) # solve the model continuously
                self.sol.array[i,:] = opt # save solutions

        return self.sol.array

    def plot_multi_par(self,x_list,y_function=None,x_lab=None,y_lab=None,show_reg=False):
        'plot H_F/H_M solution against different x values'

        # a. construct H_F/H_M fraction from solution
        frac = self.sol.array[:,3]/self.sol.array[:,1]

        # b. apply functional form if specified and transform fraction to list
        if y_function == None:
            y_list = frac.tolist()
        else:
            y_list = y_function(frac).tolist()

        # c. plot the results
        plt.scatter(x_list,y_list)

        # plot regression line if told
        if show_reg == True:
            y_list_ = [self.sol.beta0 + self.sol.beta1*x for x in x_list]
            plt.plot(x_list,y_list_)
        
        plt.grid(True) # plot settings
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)

        plt.show() # show the plot

    def run_regression(self):
        """ run regression """

        par = self.par

        x = np.log(par.wF_vec) # remember wF = 1
        A = np.vstack([np.ones(x.size),x]).T
        y = np.log(self.sol.array[:,3]/self.sol.array[:,1])
        self.sol.beta0,self.sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        beta_hat = np.array([self.sol.beta0,self.sol.beta1])

        return beta_hat