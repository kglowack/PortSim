"""
This module contains class Vessel, which, given vessel size, generates workload per hold
by different move categories. Moves are defined as either loading or discharging a 20' or 40' container. 
Used in berth and quay crane allocation simulation.

Total moves per hold follow hurdle negative binomial model. Individual move categories are then generated
using a multinomial Dirichlet regression. 
For more information see Glowacka, K., Wong, E.Y., Lam, Y.Y. (2017). Count models for vessel workload in container terminals. Unpublished manuscript

@Author: Karolina Glowacka (kglowack@gmail.com)


"""
import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.stats import fisk
import math

dist_param = pd.read_csv('Dist_Parameters.csv')  #read parameters of the distributions

class Vessel:    
    def __init__(self, name='Ship', s_type=1):
        self.name = name
        self.s_type = s_type
        #Generate number of holds for each ship type: 1=small, 2=medium, 3=large, 4=super large
        if s_type == 1:
            h_values = np.array([6,7,8,9])
            h_probs = np.array([0.01,0.04,0.4,0.54])
            self.berthing_delay = max(0,fisk.rvs(c=8.964,loc=-43.674,scale=84.868)) #for sim modeling
        elif s_type == 2:
            h_values = np.array([9,10,11,12,13,14,15,16,17,18,19])
            h_probs = np.array([0.22,0.01,0.16,0.1,0.02,0.07,0.07,0.2,0.03,0.1,0.02])
            self.berthing_delay = max(0,fisk.rvs(c=9.2918,loc=-30.274,scale=74.69 )) #for sim modeling
        elif s_type == 3:
            h_values = np.array([14,15,16,17,18,19,20,21,22])
            h_probs = np.array([0.01,0.01,0.07,0.11,0.27,0.17,0.17,0.14,0.05])
            self.berthing_delay = max(0,fisk.rvs(c=9.2672,loc=-24.342,scale=72.111)) #for sim modeling
        else:
            h_values = np.array([17,19,20,21,22])
            h_probs = np.array([0.02,0.02,0.06,0.2,0.69])
            self.berthing_delay = max(0,np.random.logistic(55.195,8.3911)) #for sim modeling
        
        self.holds = np.random.choice(h_values,1,list(h_probs))[0] + 1 #Add a dummy hold
        
        #Create data frame for workload per hold
        self.workload = pd.DataFrame(np.zeros(shape=(self.holds,5),dtype=np.int), columns = 
                                     ['Total','L20','D20','L40','D40'])
        
        #Generate workload
        for i in range(1,len(self.workload)):  #first hold always left empty
            #Genarate total number of moves per hold
            x0 = bernoulli.rvs(1-dist_param['p0'][s_type-1],size = 1)[0]
            if x0 == 0:
                self.workload.iloc[i]['Total']=0
            else:
                x1=0
                while x1==0:
                    x1_1 = np.random.gamma(dist_param['k'][s_type-1],((dist_param['k'][s_type-1]+
                                           dist_param['mu'][s_type-1] )/dist_param['k'][s_type-1])-1)
                    x1 = np.random.poisson(x1_1)
                self.workload.iloc[i]['Total']=x1
            
            #Generate individual move categories: L20 = load 20, D20 = discharge 20, L40 = load 40, D40 = discharge 40
            b1=math.exp(dist_param['LogPi1Pi4_I'][s_type-1]+dist_param['LogPi1Pi4_T'][s_type-1]*
                        self.workload.iloc[i]['Total'])
            b2=math.exp(dist_param['LogPi2Pi4_I'][s_type-1]+dist_param['LogPi2Pi4_T'][s_type-1]*
                        self.workload.iloc[i]['Total'])
            b3=math.exp(dist_param['LogPi3Pi4_I'][s_type-1]+dist_param['LogPi3Pi4_T'][s_type-1]*
                        self.workload.iloc[i]['Total'])
            
            phi = 1/(1+math.exp(-dist_param['LogitPhi'][s_type-1]))
            a0=(1-phi)/phi
            pi4 = a0/(a0*(b1+b2+b3+1))
            
            pi1=pi4*b1
            pi2=pi4*b2
            pi3=pi4*b3
            
            a1=a0*pi1
            a2=a0*pi2
            a3=a0*pi3
            a4=a0*pi4
            
            y1 = np.random.gamma(a1,4)
            y2 = np.random.gamma(a2,4)
            y3 = np.random.gamma(a3,4)
            y4 = np.random.gamma(a4,4)
            
            #Probabilities for the multinomial distribution
            p1 = y1/(y1+y2+y3+y4)
            p2 = y2/(y1+y2+y3+y4)
            p3 = y3/(y1+y2+y3+y4)
            p4 = y4/(y1+y2+y3+y4)
            
            probs = [p1,p2,p3,p4]
            
            result = np.random.multinomial(self.workload.iloc[i]['Total'],probs)
            self.workload.iloc[i]['L20'] = result[0]
            self.workload.iloc[i]['D20'] = result[1]
            self.workload.iloc[i]['L40'] = result[2]
            self.workload.iloc[i]['D40'] = result[3]
            
        self.total_workload=self.workload['Total'].sum()
        self.remaining_workload = self.total_workload
        self.total_20 = self.workload['L20'].sum() + self.workload['D20'].sum()
        self.total_40 = self.workload['L40'].sum() + self.workload['D40'].sum()
        
        self.start_space = 0  #needed for berth allocation
        self.end_space = 0    #needed for berth allocation
        
        #Needed for simulation modelling    
        self.departure_delay = max(0,np.random.lognormal(3.7911 , 0.63881 )-2.4196)
        self.t_arrival = -100    #arrival time
        self.t_berthing = -100   #time arrived at berth
        self.t_start = -100      #time start process = t_berthing + berthing_delay
        self.t_finish = -100     #time finished process
        self.t_exit = -100       #time left the port = t_finish + departure_delay
        
        self.deadline = 0     #scheduled finish time
        self.revenue = 0      #fee for port work based on the number of containers moves
        self.terminal = 0
        self.priority = self.holds * self.s_type
    
class Container:
    def __init__(self, c_type, parent):
        self.type = c_type
        self.priority = 0.0
        self.t_load = 0
        self.t_discharge = 0
        self.parent = parent
        self.parent_start_loc = -1
        self.rand_priority = 0
        self.location = 0

class Crane:
    def __init__(self, name, cap, loc, terminal):
        self.name = name
        self.cap = cap #1 = single lift, 2 = dual lift
        self.terminal = terminal
        self.location = loc #Initial location of the crane
        self.destination = 0
        self.reservation = -1
        self.direction = 0
        self.hold = False
        self.status = 'Idle'
        self.t_lastmove = 0
        self.d_lastmove = 0
        
        
        
            







