from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# plot settings
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

class RamseyModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor
        par.lambdaa = 0.5 # share of optimizing households

        # b. firms
        par.Gamma = np.nan
        par.production_function = 'ces'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'broyden' # solver for the equation system, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['Gamma','K','C','C_opt','C_htm','rk','w','r','Y','K_lag']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. find A
        ss.K = KY_ss
        Y,_,_ = production(par,1.0,ss.K)
        ss.Gamma = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.Gamma,ss.K)
        assert np.isclose(ss.Y,1.0)

        ss.r = ss.rk-par.delta
        
        # c. implied discount factor
        par.beta = 1/(1+ss.r)

        # d. consumption
        ss.C = ss.Y - par.delta*ss.K
        ss.C_htm = ss.w*(1-par.lambdaa)
        ss.C_opt = ss.C - ss.C_htm

        if do_print:
            
            varnames = ['Y_ss','K_ss/Y_ss','rk_ss','r_ss','w_ss','Gamma','beta','C_opt_ss','C_htm_ss','C_ss']
            varvalues = [ss.Y,ss.K/ss.Y,ss.rk,ss.r,ss.w,ss.Gamma,par.beta,ss.C_opt,ss.C_htm,ss.C]

            for name,value in zip(varnames,varvalues):
                print(f'{name:10s} = {value:.4f}')

    def evaluate_path_errors(self,do_print=False):
        """ evaluate errors along transition path """

        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption for optimizing households     
        C_opt = path.C_opt
        C_opt_plus = np.append(path.C_opt[1:],ss.C_opt)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        
        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.Gamma,K_lag)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)

        # d. consumption for hand-to-mouth households
        path.C_htm = path.w*(1-par.lambdaa)

        # e. total consumption
        path.C = path.C_htm + C_opt

        # f. errors (also called H)
        errors = np.nan*np.ones((2,par.Tpath))
        errors[0,:] = C_opt**(-par.sigma) - par.beta*(1+r_plus)*C_opt_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - path.C_htm - C_opt))

        if do_print==True:
            print(f'Error_euler: {np.max(errors[0,:])}')
            print(f'Error_capital: {np.max(errors[1,:])}')
        
        return errors.ravel()
        
    def calculate_jacobian(self,h=1e-6):
        """ calculate jacobian """
        
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. allocate
        Njac = 2*par.Tpath
        jac = self.jac = np.nan*np.ones((Njac,Njac))
        
        x_ss = np.nan*np.ones((2,par.Tpath))
        x_ss[0,:] = ss.C_opt
        x_ss[1,:] = ss.K
        x_ss = x_ss.ravel()

        # b. baseline errors
        path.K[:] = ss.K
        path.C_opt[:] = ss.C_opt
        base = self.evaluate_path_errors()

        # c. jacobian
        for i in range(Njac):
            
            # i. add small number to a single x (single K or C_opt) 
            x_jac = x_ss.copy()
            x_jac[i] += h
            x_jac = x_jac.reshape((2,par.Tpath))
            
            # ii. alternative errors
            path.K[:] = x_jac[1,:]            
            path.C_opt[:] = x_jac[0,:]
            alt = self.evaluate_path_errors()

            # iii. numerical derivative
            jac[:,i] = (alt-base)/h
        
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((2,par.Tpath))
            path.C_opt[:] = x[0,:]
            path.K[:] = x[1,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess
        x0 = np.nan*np.ones((2,par.Tpath))
        x0[0,:] = ss.C_opt
        x0[1,:] = ss.K
        x0 = x0.ravel()

        # c. call solver
        if par.solver == 'broyden':

            x = broyden_solver(eq_sys,x0,self.jac,do_print=do_print)
        
        elif par.solver == 'scipy':
            
            root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
            # the factor determines the size of the initial step
            #  too low: slow
            #  too high: prone to errors
             
            x = root.x

        else:

            raise NotImplementedError('unknown solver')
            
        # d. final evaluation
        eq_sys(x)

    def big_plot(self):
        """ plot transition paths for variables """

        # plot setup
        fig = plt.figure(figsize=(15,10))
        
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax4 = fig.add_subplot(2,3,4)                
        ax5 = fig.add_subplot(2,3,5)

        ax1.set_title(r'$r_t$')
        ax2.set_title(r'$w_t$')
        ax3.set_title(r'$C_t$')
        ax4.set_title(r'$Y_t$')
        ax5.set_title(r'$K_t$')

        end = 120 # plot periods

        ax1.set_xlim(0,end)
        ax2.set_xlim(0,end)
        ax3.set_xlim(0,end)
        ax4.set_xlim(0,end)
        ax5.set_xlim(0,end) 
        
        ax1.plot(self.path.r)
        ax2.plot(self.path.w)
        ax3.plot(self.path.C_opt,label=r'$C_t^{Opt}$')
        ax3.plot(self.path.C_htm,label=r'$C_t^{Htm}$')
        ax4.plot(self.path.Y)
        ax5.plot(self.path.K)

        ax3.legend()        

    def simulate_and_plot(self,parameter,values):
        """ simulate the model for different parameter values and plots paths """

        # plot setup
        fig = plt.figure(figsize=(15,10))
        
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax4 = fig.add_subplot(2,3,4)                
        ax5 = fig.add_subplot(2,3,5)

        ax1.set_title(r'$r_t$')
        ax2.set_title(r'$w_t$')
        ax3.set_title(r'$C_t$')
        ax4.set_title(r'$Y_t$')
        ax5.set_title(r'$K_t$')

        end = 120 # plot periods

        ax1.set_xlim(0,end)
        ax2.set_xlim(0,end)
        ax3.set_xlim(0,end)
        ax4.set_xlim(0,end)
        ax5.set_xlim(0,end)

        colors = ['blue', 'green', 'red', 'orange', 'purple']

        # loop over parameter values
        for i,value in enumerate(values):

            # a. set parameter value
            setattr(self.par,parameter,value)

            # b. find steady state
            self.find_steady_state(KY_ss=4.0,do_print=False)

            # c. calculate jacobian
            self.calculate_jacobian()

            # d. solve
            self.solve(do_print=False)

            # e. plot
            color = colors[i % len(colors)] 

            ax1.plot(self.path.r,color=color)
            ax2.plot(self.path.w,color=color)
            ax3.plot(self.path.C_opt,color=color)
            ax3.plot(self.path.C_htm,color=color,linestyle='--')
            ax4.plot(self.path.Y,color=color)
            ax5.plot(self.path.K,color=color)

            # custom legend
            handles = [Line2D([0], [0], color=colors[j], label='Value = '+str(value)) for j,value in enumerate(values)]
            handles.append(Line2D([0], [0], color='black', linestyle='--', label=r'$C_t^{Htm}$'))
            handles.append(Line2D([0], [0], color='black', linestyle='-',  label=r'$C_t^{Opt}$'))

            labels = [h.get_label() for h in handles]

            fig.legend(handles=handles, labels=labels, loc='lower right',bbox_to_anchor=(0.85, 0.275))

def production(par,Gamma,K_lag):
    """ production and factor prices """

    # a. production and factor prices
    if par.production_function == 'ces':

        # a. production
        Y = Gamma*( par.alpha*K_lag**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # b. factor prices
        rk = Gamma*par.alpha*K_lag**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)
        w = Gamma*(1-par.alpha)*(1.0)**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # a. production
        Y = Gamma*K_lag**par.alpha * (1.0)**(1-par.alpha)

        # b. factor prices
        rk = Gamma*par.alpha * K_lag**(par.alpha-1) * (1.0)**(1-par.alpha)
        w = Gamma*(1-par.alpha) * K_lag**(par.alpha) * (1.0)**(-par.alpha)

    else:

        raise Exception('unknown type of production function')

    return Y,rk,w            

def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False):
    """ numerical equation system solver using the broyden method 
    
        f (callable): function return errors in equation system
        jac (ndarray): initial jacobian
        tol (float,optional): tolerance
        maxiter (int,optional): maximum number of iterations
        do_print (bool,optional): print progress

    """

    # a. initial
    x = x0.ravel()
    y = f(x)

    # b. iterate
    for it in range(maxiter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print: print(f' it = {it:3d} -> max. abs. error = {abs_diff:12.8f}')

        if abs_diff < tol: return x
        
        # ii. new x
        dx = np.linalg.solve(jac,-y)
        assert not np.any(np.isnan(dx))
        
        # iii. evaluate
        ynew = f(x+dx)
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        x += dx
            
    else:

        raise ValueError(f'no convergence after {maxiter} iterations')        