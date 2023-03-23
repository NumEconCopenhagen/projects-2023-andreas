
from types import SimpleNamespace

import numpy as np
from scipy import optimize

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

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*x[0] + par.wF*x[2]

        # b. home production
        if par.sigma == 0.0:
            H = np.min(x[1],x[3])

        elif par.sigma == 1.0:
            H = x[1]**(1-par.alpha)*x[3]**par.alpha

        else:
            exponent = (par.sigma-1)/par.sigma
            term1 = (1 - par.alpha) * x[1]**exponent
            term2 = par.alpha * x[3]**exponent
            H = (term1 + term2)**(1/exponent)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = x[0]+x[1]
        TF = x[2]+x[3]
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
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

        par = self.par
        sol = self.sol

        LM = np.nan # vector
        HM = np.nan
        LF = np.nan
        HF = np.nan

        varlist = [LM, HM, LF, HF]
        x = np.array([LM, HM, LF, HF])

        # a. contraint function (negative if violated)
        constraints = ({'type': 'ineq', 'fun': lambda x: 24-x[0]-x[1]},
                       {'type': 'ineq', 'fun': lambda x: 24-x[2]-x[3]})
        bounds = [(0,24),(0,24),(0,24),(0,24)]

        # b. call optimizer
        initial_guess = [1,1,1,1] # some guess, should be feasible
        res = optimize.minimize(
            self.calc_utility_, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints)

        # print(res.message) # check that the solver has terminated correctly

        return res.x

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass

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

    def plot_multi_par(self,x_list,y_function=None,x_lab=None,y_lab=None):
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
        
        plt.grid(True) # plot settings
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)

        plt.show() # show the plot
         