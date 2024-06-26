import numpy as np
from scipy import optimize
from scipy.optimize import differential_evolution, LinearConstraint, Bounds
from types import SimpleNamespace
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
        par.epsilonM = 1.0
        par.epsilonF = 1.0
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

    def calc_utility(self,LM,HM,LF,HF,expand=False,estimate=False):
        """ calculate utility """

        par = self.par

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0.0:
            if estimate == True:
                H = min(HM,HF) # optimizer does not like np.min?
            else:
                H = np.min(HM,HF)

        elif par.sigma == 1.0:
            H = HM**(1-par.alpha)*HF**par.alpha

        else:
            exponent = (par.sigma-1)/par.sigma
            term1 = (1 - par.alpha) * HM**exponent
            term2 = par.alpha * HF**exponent
            H = (term1 + term2)**(1/exponent)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        
        if estimate == True:
            utility = max(Q,1e-8)**(1-par.rho)/(1-par.rho) # optimizer does not like np.fmax?
        else:
            utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work

        # d1. total hours
        TM = LM+HM
        TF = LF+HF

        # d2. seperate utility from house work and market work (expansion in question 5)
        epsilon_ = 1+1/par.epsilon
        epsilonM_ = 1+1/par.epsilonM
        epsilonF_ = 1+1/par.epsilonF
        disutilityM = HM**epsilonM_/epsilonM_+LM**epsilon_/epsilon_
        disutilityF = HF**epsilonF_/epsilonF_+LF**epsilon_/epsilon_

        # d3. total disutility of work
        if expand == True:
            disutility = par.nu*(disutilityM + disutilityF)
        else:  
            disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """

        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(1e-8,24,49)
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

    def solve(self,expand,estimate=False):
        """ solve model continuously """

        # a. contraint function
        lc = LinearConstraint([[1,1,0,0],[0,0,1,1]],[0,0],[24,24])

        # b. wrapper function
        def calc_utility_(x):                 
            return -self.calc_utility(x[0],x[1],x[2],x[3],expand,estimate)

        # b. call optimizer
        res = optimize.minimize(calc_utility_,
                                x0=[5,5,5,5],
                                method='BFGS',
                                bounds=[(0,24),(0,24),(0,24),(0,24)],constraints=lc)

        return res.x

    def solve_multi_par(self,par_name,par_list,expand,discrete,estimate=False):
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
                opt = self.solve(expand,estimate) # solve the model
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

    def estimate(self,expand):
        """ estimate alpha and sigma """

        # a. output of this function to be minimized
        def squared_dev(x):

            # 1. set alpha and beta
            setattr(self.par,'sigma',x[0])
            setattr(self.par,'alpha',x[1])
            
            # 2. true beta0 and beta1
            beta = np.array([self.par.beta0_target,
                             self.par.beta1_target])

            # 3. list of wages to solve model for
            wages = [0.8,0.9,1.0,1.1,1.2]

            # 4. solve the model for different wages
            self.solve_multi_par('wF',wages,expand,estimate=True,discrete=False)

            # 5. run regression and save output
            beta_hat = self.run_regression()
            self.sol.beta0 = beta_hat[0]
            self.sol.beta1 = beta_hat[1]

            return np.sum((beta - beta_hat) ** 2)

        # b. call optimizer
        res = optimize.minimize(squared_dev,
                                x0=[0.75,0.75],
                                method='Nelder-Mead',
                                bounds=[(0,2),(0,1)],tol=0.01)

        return res

    def run_regression(self):
        """ run regression """

        par = self.par

        x = np.log(par.wF_vec) # remember wF = 1
        A = np.vstack([np.ones(x.size),x]).T
        y = np.log(self.sol.array[:,3]/self.sol.array[:,1])
        self.sol.beta0,self.sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        beta_hat = np.array([self.sol.beta0,self.sol.beta1])

        return beta_hat