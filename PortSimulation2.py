"""
Author: Karolina Glowacka (kglowack@gmail.com)
In the following model, I modified the port configuration in order to protect the identity
of the studied port operator. The following assumptions are made:
- Two terminals: 
    T1 - 475m long, divided into 38 12.5m long segments, corresponding to vessel holds
    T2 - 550m long, divided into 44 12.5m long segments
- Total number of cranes = 30
- Number of cranes at T1: 14, including 6 dual lift
- Number of cranes at T2: 16, including 11 dual lift
- Each crane requires 25m of berth space to operate, therefore it will be assigned to two
vessel holds at the same time. For crane operation purposes, the terminals are divided into
25m long segments, representing crane work zones (19 zones for T1 and 22 zones for T2).

Berth space is allocated using best-fit bin packing method. The berthing queue discipline
can be one of the three options: FIFO, Descending (based on ship length x category), and
SPT (approximation of the shortest processing time by considering total number of container
moves in a given vessel).

Crane allocation is decided on the basis of work priority in each crane zone, calculated as the
workload in a particular zone (consisting of two vessel holds) divided by the total remaining
workload of the vessel. Workload is measured in terms of the number of container moves. Crane
allocation decisions (reassignment) are made under the following circumstances:
1. A new vessel arrives and is ready for operations (global within the affected terminal)
2. A vessel completes the operations (global within the affected terminal)
3. A crane finishes all work in its assigned zone (local)
4. Intermittent crane allocation every hour (global for both terminals)
"""

import vessel_sim2 as v
import numpy as np
import pandas as pd
import simpy
import csv
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from joblib import Parallel, delayed
import multiprocessing
import os
import sys

#Recipe from pyinstaller wiki page for multiprocessing on windows
#https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
try:
    if sys.platform.startswith('win'):
        import multiprocessing.popen_spawn_win32 as forking
    else:
        import multiprocessing.popen_fork as forking
except ImportError:
    import multiprocessing.forking as forking

if sys.platform.startswith('win'):
    class _Popen(forking.Popen):
        def __init__(self, *args, **kw):
            if hasattr(sys, 'frozen'):
                os.putenv('_MEIPASS2', sys._MEIPASS)
            try:
                super(_Popen, self).__init__(*args, **kw)
            finally:
                if hasattr(sys, 'frozen'):
                    if hasattr(os, 'unsetenv'):
                        os.unsetenv('_MEIPASS2')
                    else:
                        os.putenv('_MEIPASS2', '')

    forking.Popen = _Popen

    try:
        from ctypes import *

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

# GLOBAL VARIABLE DECLARATION (unchangeable):

CAT = [1, 2, 3, 4] #ship size categories: 1=small, 2=medium, 3=large, 4=super large
PROB = [0.134380454, 0.5008726, 0.253054101, 0.111692845] #probabilities of ships belonging to each category

T1_SIZE = 38
T2_SIZE = 44

TOT_SPACE = T1_SIZE + T2_SIZE

#Lists for crane information: [starting location, end location, number of cranes, first crane in the terminal]
w1 = [0, T1_SIZE // 2, 14, 1]
w2 = [T1_SIZE // 2, 41, 16, 15]

term_dic = {1: w1, 2: w2}

WARMUP = 24 * 60 # Warmup period in the simulation is assumed to be 1 day.

#CLASS DEFINITIONS

class SimVars: #Define simulation variables (read through the GUI)
    def __init__(self):
        self.filename = 'results2.csv'
        self.REPLICATIONS = 100
        self.berth_discipline = 'fifo'
        self.cores = 1
        self.RUN = True
        self.SIM_TIME = 30*1440

class Port:
    def __init__(self, env):
        self.env = env
        self.dummy_m = simpy.Resource(env, 100)
        self.crane_res = []
        for i in range(41):
            a = simpy.Resource(env, 1) #A resource is defined for each crane zone
            self.crane_res.append(a)
        self.BERTH_QUEUE = []
        self.WORK = []
        self.IN_PROCESS = []
        self.COMPLETED = []
        self.Crane_List = []
        #Create a data frame representing vessel allocation at the berth
        self.TERMINAL = pd.DataFrame(np.zeros(shape=(TOT_SPACE,3),dtype=np.int), columns = ['Name','Space','Terminal'])
        #Create a data frame representing crane locations and workload
        self.CRANES = pd.DataFrame(np.zeros(shape=(TOT_SPACE//2,4)),columns = ['Terminal', 'Priority', 'Cranes', 'Moves'])
        self.rep = 0
        self.filename = ''

    def dummy(self,ships):
        yield self.env.timeout(0)


# FUNCTION DEFINITIONS:

def run_GUI(g):
    num_cores = multiprocessing.cpu_count()
    def callbackOK2(event):
        g.berth_discipline = bd.get()
        g.filename = file_entry.get()
        if len(g.filename) == 0:
            g.filename = 'results' + '_' + g.berth_discipline + '.csv'
        elif g.filename[-4:] != '.csv':
            g.filename = g.filename + '_' + g.berth_discipline + '.csv'
        else:
            g.filename = g.filename[:-4] + '_' + g.berth_discipline + '.csv'
        g.REPLICATIONS = int(rep_entry.get())
        g.SIM_TIME = (float(length_entry.get()) + 1) * 60 * 24 #Add one day for warmup

        g.cores = int(core_entry.get())
        root.destroy()

    def callbackOK():
        g.berth_discipline = bd.get()
        g.filename = file_entry.get()
        if len(g.filename) == 0:
            g.filename = 'results' + '_' + g.berth_discipline + '.csv'
        elif g.filename[-4:] != '.csv':
            g.filename = g.filename + '_' + g.berth_discipline + '.csv'
        else:
            g.filename = g.filename[:-4] + '_' + g.berth_discipline + '.csv'
        g.REPLICATIONS = int(rep_entry.get())
        g.SIM_TIME = (float(length_entry.get())+1) * 60 * 24 #Add one day for warmup

        g.cores = int(core_entry.get())
        root.destroy()

    def callbackCancel():
        g.RUN = False
        root.destroy()

    def checkReps(event):
        try:
            a = int(rep_entry.get())
        except:
            messagebox.showinfo("Wrong Input", "Number of replications has to be an integer")
            rep_entry.focus()
    def checkCores(event):
        try:
            a = int(core_entry.get())
        except:
            messagebox.showinfo("Wrong Input", "Number of cores has to be an integer")
            core_entry.focus()
        if a > num_cores:
            messagebox.showinfo("Warning", "You specified more cores than what's available in your system")
            core_entry.focus()
    def checkSimTime(event):
        try:
            a = float(length_entry.get())
        except:
            messagebox.showinfo("Wrong Input", "Simulation time has to be numeric")
            length_entry.focus()

    def onClose():
        g.RUN = False
        root.destroy()

    root = Tk()
    root.title(' Port Simulation')

    topframe = ttk.Frame(root, padding="12")
    topframe.grid(column=0, row=1, sticky=(N, W, E, S))
    topframe.columnconfigure(0, weight=1)
    topframe.rowconfigure(1, weight=1)

    mainframe = ttk.Frame(root, padding="12")
    mainframe.grid(column=0, row=2, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(2, weight=1)

    buttonframe = ttk.Frame(root, padding="12")
    buttonframe.grid(column=0, row=4, sticky=(N, W, E, S))
    buttonframe.columnconfigure(0, weight=1)
    buttonframe.rowconfigure(4, weight=1)

    coreframe = ttk.Frame(root, padding="12")
    coreframe.grid(column=0, row=3, sticky=(N, W, E, S))
    coreframe.columnconfigure(0, weight=1)
    coreframe.rowconfigure(3, weight=1)

    ttk.Label(topframe, text='Berth priority:').grid(column=0, row=1, sticky=W)

    reps = IntVar(topframe, value=100)
    ttk.Label(topframe, text='No. replications:').grid(column=1, row=2, sticky=W)
    rep_entry = ttk.Entry(topframe, width=5, textvariable=reps)
    rep_entry.grid(column=2, row=2, sticky=(W))

    length = IntVar(topframe, value=30)
    ttk.Label(topframe, text='Replication length: ').grid(column=1, row=3, sticky=W)
    length_entry = ttk.Entry(topframe, width=5, textvariable=length)
    length_entry.grid(column=2, row=3, sticky=(W))
    ttk.Label(topframe, text='days').grid(column=3, row=3, sticky=W)

    bd = StringVar(topframe, value='fifo')
    fifo = ttk.Radiobutton(topframe, text='FIFO', variable=bd, value='fifo')
    fifo.grid(column=0, row=2, sticky=W)
    desc = ttk.Radiobutton(topframe, text='Descending', variable=bd, value='desc')
    desc.grid(column=0, row=3, sticky=W)
    desc = ttk.Radiobutton(topframe, text='SPT', variable=bd, value='spt')
    desc.grid(column=0, row=4, sticky=W)

    ttk.Label(mainframe, text='Enter output file name:').grid(column=0, row=1, sticky=W)
    file_entry = ttk.Entry(mainframe, width=20, textvariable=g.filename)
    file_entry.grid(column=1, row=1, sticky=(W))
    ttk.Label(mainframe, text=".csv").grid(column=2, row=1, sticky=W)

    c = IntVar(coreframe, value=num_cores)
    ttk.Label(coreframe, text="Number of cores to use: ").grid(column=0, row=1, sticky=W)
    core_entry = ttk.Entry(coreframe, width=3, textvariable=c)
    core_entry.grid(column=1, row=1, sticky=(W))
    ttk.Label(coreframe, text='max   =').grid(column=2, row=1, sticky=E)
    ttk.Label(coreframe, text=num_cores).grid(column=3, row=1, sticky=W)

    ttk.Button(buttonframe, text="OK", command=callbackOK).grid(column=0, row=1, sticky=E)
    ttk.Button(buttonframe, text="Cancel", command=callbackCancel).grid(column=1, row=1, sticky=E)

    for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
    for child in topframe.winfo_children(): child.grid_configure(padx=5, pady=2)
    for child in coreframe.winfo_children(): child.grid_configure(padx=5, pady=1)
    for child in buttonframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    file_entry.focus()
    root.bind('<Return>', callbackOK2)
    rep_entry.bind('<FocusOut>',checkReps)
    core_entry.bind('<FocusOut>', checkCores)
    length_entry.bind('<FocusOut>', checkSimTime)

    root.protocol("WM_DELETE_WINDOW", onClose)

    root.mainloop()

def ship_arrival(env,name,port,g):
    ship_type = np.random.choice(CAT,1,list(PROB))[0] #Generate vessel category
    ship = v.Vessel(name=name, s_type=ship_type) #Create a vessel object
    port.BERTH_QUEUE.append(ship) #Add vessel to the berthing queue
    #Sort berthing queue according to the discipline selected
    if g.berth_discipline == 'desc':
        port.BERTH_QUEUE.sort(key=lambda priority: ship.priority, reverse=True)
    elif g.berth_discipline == 'spt':
        port.BERTH_QUEUE.sort(key=lambda priority: ship.total_workload)
    ship.t_arrival = env.now #Record vessel arrival time
    with port.dummy_m.request() as request: #Use dummy resource in SimPy (needed to make the code work)
        yield request
    yield env.process(port.dummy(name))
    current = release_ship(env,port) #Check if any vessel can be assigned berth space
    while current != -1:
        env.process(start_operations(env,current,port))
        current = release_ship(env,port) #Check if another vessel can be moved to a berth

def release_ship(env,port): #Select a ship from the berthing queue to allocate to a berth space
    selected=-1
    max_spaces = port.TERMINAL['Space'].max()
    for item in port.BERTH_QUEUE:
        if max_spaces >= item.holds:
            selected=item
            break
    if selected != -1:
        selected.t_berthing = env.now
        port.BERTH_QUEUE.pop(port.BERTH_QUEUE.index(selected))
        port.IN_PROCESS.append(selected)
        #start best fit
        best_space = 1000
        best_index = 0
        for i in range(TOT_SPACE):
            spaces_before = 0
            if port.TERMINAL.iloc[i]['Space'] == selected.holds:
                for k in range(i-1,-1,-1):
                    if k>=0 and port.TERMINAL.iloc[k]['Terminal']==port.TERMINAL.iloc[k+1]['Terminal'] and port.TERMINAL.iloc[k]['Space']>0:
                        spaces_before += 1
                    else:
                        break
                if spaces_before < best_space:
                    best_space = spaces_before
                    best_index = i
        selected.start_space = best_index
        selected.end_space = best_index + selected.holds - 1
        selected.terminal = port.TERMINAL.iloc[best_index]['Terminal']
        for i in range(best_index,selected.end_space+1):
            port.TERMINAL.iloc[i]['Space']=0
            port.TERMINAL.iloc[i]['Name']=selected.holds
        for k in range(best_index-1,-1,-1):
            if k>=0 and port.TERMINAL.iloc[k]['Terminal']==port.TERMINAL.iloc[k+1]['Terminal'] and port.TERMINAL.iloc[k]['Space']>0:
                port.TERMINAL.iloc[k]['Space']=port.TERMINAL.iloc[k+1]['Space']+1
            else:
                break
    return selected


def start_operations(env,ship,port):
    yield env.timeout(ship.berthing_delay)
    ship.t_start = env.now
    generate_containers(env,ship,port)
    yield env.timeout(0.1)
#    crane_control(env,port)
 #   yield env.timeout(0.1)
    crane_reassign(env,ship.terminal,port)


def generate_containers(env,parent,port):
    for i in range(parent.holds):
        for k in range(parent.workload.iloc[i]['L20']):
            container = v.Container(c_type = 'L20', parent=parent)
            container.rand_priority = np.random.uniform()+1
            port.WORK[(parent.start_space+i)//2].append(container)
            container.location = (parent.start_space+i)//2
        for k in range(parent.workload.iloc[i]['D20']):
            container = v.Container(c_type = 'D20', parent=parent)
            container.rand_priority = np.random.uniform()
            port.WORK[(parent.start_space+i)//2].append(container)
            container.location = (parent.start_space+i)//2
        for k in range(parent.workload.iloc[i]['L40']):
            container = v.Container(c_type = 'L40', parent=parent)
            container.rand_priority = np.random.uniform()+1
            port.WORK[(parent.start_space+i)//2].append(container)
            container.location = (parent.start_space+i)//2
        for k in range(parent.workload.iloc[i]['D40']):
            container = v.Container(c_type = 'D40', parent=parent)
            container.rand_priority = np.random.uniform()
            port.WORK[(parent.start_space+i)//2].append(container)
            container.location = (parent.start_space+i)//2

    start = parent.start_space // 2
    end = parent.end_space // 2
    for i in range(start, end+1):
        if len(port.WORK[i]) != 0:
            port.WORK[i].sort(key=lambda x: x.rand_priority)
            port.CRANES.iloc[i]['Priority'] = (len(port.WORK[i])/port.WORK[i][0].parent.remaining_workload)*port.WORK[i][0].parent.s_type
            port.CRANES.iloc[i]['Moves']=len(port.WORK[i])
        else:
            port.CRANES.iloc[i]['Priority'] = 0
            port.CRANES.iloc[i]['Moves']=0
      
def crane_operation(env,crane,port):
#    print('Crane op at t = ', env.now, ' crane: ', crane.name)
    if crane.status == 'Idle' and crane.hold == False and len(port.WORK[crane.location])>0:
        crane.status = 'Busy'
        env.process(move_container(env,crane,port))

def move_container(env,crane,port):
    moved = 1
    cont = None
    for i in range(len(port.WORK)):
        if len(port.WORK[i]) > 0:
            port.CRANES.iloc[i]['Priority']=len(port.WORK[i])/port.WORK[i][0].parent.remaining_workload
            port.CRANES.iloc[i]['Moves']=len(port.WORK[i])
        else:
            port.CRANES.iloc[i]['Priority']=0
            port.CRANES.iloc[i]['Moves']=0 
    if len(port.WORK[crane.location]) > 0:
        parent = port.WORK[crane.location][0].parent
        crane.status = 'Busy'
        if crane.cap == 2 and len(port.WORK[crane.location]) >= 2:
            if (port.WORK[crane.location][0].type == 'L20' and port.WORK[crane.location][1].type == 'L20') or (port.WORK[crane.location][0].type == 'D20' and port.WORK[crane.location][1].type == 'D20'):
                cont = port.WORK[crane.location].pop(0)
                cont = port.WORK[crane.location].pop(0)
                moved = 2
                a = np.random.triangular(3.05, 3.15, 3.25)
            else:
                cont = port.WORK[crane.location].pop(0)
                moved = 1
                a = np.random.triangular(2.3, 2.4, 2.5)
        else:
            cont = port.WORK[crane.location].pop(0)
            moved = 1
            a = np.random.triangular(2.3, 2.4, 2.5)
        with port.crane_res[crane.location].request() as request:
            yield request
            crane.t_lastmove = env.now
            crane.d_lastmove = a
#            print('Removing ', moved, ' from index ', crane.location, ' with crane ', crane.name, ' t= ', env.now, ' left = ', len(port.WORK[crane.location]))
            yield env.timeout(a)
    #    print('index is ', crane.location, ' with workload = ', len(WORK[crane.location]))
        parent.remaining_workload = parent.remaining_workload-moved #update remianing workload of the parent vessel
        if parent.remaining_workload == 0:
            env.process(ship_departure(env, parent, port))
            port.CRANES.iloc[crane.location]['Priority']=0
            crane.status = 'Idle'
        else:
            port.CRANES.iloc[crane.location]['Priority']=float(len(port.WORK[crane.location]))/float(parent.remaining_workload)
        port.CRANES.iloc[crane.location]['Moves']=len(port.WORK[crane.location])
    #    print('item in resource ', crane.location, ' = ', port.crane_res[crane.location].users)
#    crane.status = 'Idle'
    if len(port.WORK[crane.location]) == 0:
        port.CRANES.iloc[crane.location]['Priority'] = 0
        port.CRANES.iloc[crane.location]['Moves'] = 0
        crane.status = 'Idle'
        local_reassign(env,parent.terminal,crane,port)
    elif not crane.hold:
        env.process(move_container(env,crane,port))    

def crane_reassign(env,t,port):
    for i in range(len(port.WORK)):
        if len(port.WORK[i]) > 0:
            port.CRANES.iloc[i]['Priority']=(len(port.WORK[i])/port.WORK[i][0].parent.remaining_workload)*port.WORK[i][0].parent.s_type
            port.CRANES.iloc[i]['Moves']=len(port.WORK[i])
        else:
            port.CRANES.iloc[i]['Priority']=0
            port.CRANES.iloc[i]['Moves']=0        
    cont = True
    priorities = []
    a = int(term_dic.get(t)[0])
    b = int(term_dic.get(t)[1])
    for i in range(a, b):
        priorities.append((port.CRANES.iloc[i]['Priority'],i))
    priorities.sort(reverse=True)
    if priorities[0][0] == 0:
        cont = False
    if cont:
        prop = priorities[:term_dic.get(t)[2]]
        prop.sort(key=lambda x: x[1])
        for i in range(term_dic.get(t)[2]):
            port.Crane_List[i+term_dic.get(t)[3]-1].destination = prop[i][1]
            if port.Crane_List[i+term_dic.get(t)[3]-1].destination != port.Crane_List[i+term_dic.get(t)[3]-1].location:
                if port.Crane_List[i+term_dic.get(t)[3]-1].destination > port.Crane_List[i+term_dic.get(t)[3]-1].location:
                    port.Crane_List[i+term_dic.get(t)[3]-1].direction = 1
                else:
                    port.Crane_List[i+term_dic.get(t)[3]-1].direction = -1
            else:
                port.Crane_List[i+term_dic.get(t)[3]-1].direction = 0
        for i in range(term_dic.get(t)[2]):
            env.process(move_crane(env,port.Crane_List[i+term_dic.get(t)[3]-1],port))

    else:
        for i in range(term_dic.get(t)[2]):
#            port.Crane_List[i+term_dic.get(t)[3]-1].status = 'Idle'
            port.Crane_List[i+term_dic.get(t)[3]-1].hold = False
            port.Crane_List[i+term_dic.get(t)[3]-1].direction = 0
#            crane_operation(env,Crane_List[i+term_dic.get(t)[3]-1],port)

    
def move_crane(env,crane,port):
    moved = False
    while crane.destination != crane.location:
        crane.hold = True
        moved = True
        crane.reservation = crane.destination
        if crane.status == 'Busy':
            yield env.timeout(max(0,(crane.d_lastmove+crane.t_lastmove - env.now)+0.0001))
        crane.status = 'Moving'
        while port.CRANES.iloc[crane.location+crane.direction]['Cranes']!=0:
#            print('Waiting for crane ', port.CRANES.iloc[crane.location+crane.direction]['Cranes'])
            yield env.timeout(1)
        if port.CRANES.iloc[crane.location+crane.direction]['Cranes']==0:
#            print('Moving Crane: ', crane.name, ' from ', crane.location, ' to ', crane.location+crane.direction, ' ', env.now)
            yield env.timeout(1)
            port.CRANES.iloc[crane.location]['Cranes']=0
            crane.location = crane.location + crane.direction
            port.CRANES.iloc[crane.location]['Cranes']=crane.name    
    if moved:
        crane.reservation = -1
        crane.status = 'Idle'
        crane.hold = False
        crane.direction = 0
        crane_operation(env,crane,port)
    elif crane.status == 'Idle':
        crane_operation(env,crane,port)
    


def local_reassign(env,t,crane,port):
    for i in range(len(port.WORK)):
        if len(port.WORK[i]) > 0:
            port.CRANES.iloc[i]['Priority']=(len(port.WORK[i])/port.WORK[i][0].parent.remaining_workload)*port.WORK[i][0].parent.s_type
            port.CRANES.iloc[i]['Moves']=len(port.WORK[i])
        else:
            port.CRANES.iloc[i]['Priority']=0
            port.CRANES.iloc[i]['Moves']=0
#    print('Local moving crane ', crane.name, ' terminal ', t)
#    print('Running local')
    if crane.name == term_dic.get(t)[3]: #if first crane in the terminal
        start = term_dic.get(t)[0]
        if (port.Crane_List[crane.name].destination < port.Crane_List[crane.name].location) and (port.Crane_List[crane.name].destination > crane.location):
            end = port.Crane_List[crane.name].destination
        else:
            end = port.Crane_List[crane.name].location
    elif crane.name == term_dic.get(t)[3]+term_dic.get(t)[2]-1: #if last crane in the terminal
        if (port.Crane_List[crane.name-2].destination > port.Crane_List[crane.name-2].location):
            start = port.Crane_List[crane.name-2].destination+1
        else:
            start = port.Crane_List[crane.name-2].location+1
        end = term_dic.get(t)[1]
    else:
        if (port.Crane_List[crane.name-2].destination > port.Crane_List[crane.name-2].location):
            start = port.Crane_List[crane.name-2].destination+1
        else:
            start = port.Crane_List[crane.name-2].location+1
        if (port.Crane_List[crane.name].destination < port.Crane_List[crane.name].location) and (port.Crane_List[crane.name].destination > crane.location):
            end = port.Crane_List[crane.name].destination
        else:
            end = port.Crane_List[crane.name].location
    max_p = port.CRANES.iloc[start]['Priority']
    max_i = start
    if start != end:
        for i in range(start+1, end):
            if port.CRANES.iloc[i]['Priority'] > max_p:
                max_p = port.CRANES.iloc[i]['Priority']
                max_i = i
    if max_p != 0:
        crane.destination = max_i
        if crane.destination != crane.location:
            if crane.destination > crane.location:
                crane.direction = 1
            else:
                crane.direction = -1
            env.process(move_crane(env,crane,port)) 
        else:
            crane.direction = 0

def intermittent_reassign(env,port):
    while True:
        if env.now >= 60:
            crane_reassign(env,1,port)
            crane_reassign(env,2,port)
        yield env.timeout(60)
   

def ship_departure(env,ship,port):
    ship.t_finish = env.now
    port.IN_PROCESS.pop(port.IN_PROCESS.index(ship))
    yield env.timeout(ship.departure_delay)
    ship.t_exit = env.now
    port.COMPLETED.append(ship)
    if True: #ship.t_arrival > WARMUP:
        with open(port.filename,'a',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([port.rep+1, ship.name, ship.s_type, ship.holds, ship.terminal,
                             ship.total_workload, ship.total_20, ship.total_40, ship.t_arrival,
                         ship.t_berthing, ship.t_start, ship.t_finish, ship.t_exit])

    #Update TERMINAL['Space']
    if ship.end_space < TOT_SPACE - 1:
        if ship.terminal == port.TERMINAL.iloc[ship.end_space+1]['Terminal']:
            port.TERMINAL.iloc[ship.end_space]['Space']=port.TERMINAL.iloc[ship.end_space+1]['Space']+1
        else:
            port.TERMINAL.iloc[ship.end_space]['Space']=1
    else:
        port.TERMINAL.iloc[ship.end_space]['Space']=1
    port.TERMINAL.iloc[ship.end_space]['Name']=0
    for i in range(ship.end_space-1,ship.start_space-1,-1):
        port.TERMINAL.iloc[i]['Space']=port.TERMINAL.iloc[i+1]['Space']+1
        port.TERMINAL.iloc[i]['Name']=0
    for i in range(ship.start_space,0,-1):
        if ship.terminal == port.TERMINAL.iloc[i-1]['Terminal']:
            if port.TERMINAL.iloc[i-1]['Space'] > 0:
                port.TERMINAL.iloc[i-1]['Space']=port.TERMINAL.iloc[i]['Space']+1
        else:
            break
    current = release_ship(env,port)
    while current != -1:
        env.process(start_operations(env,current,port))
        current = release_ship(env,port)
    else:
        crane_reassign(env, ship.terminal,port)
        
   
def setup(env,port,g):
    i = 1
    yield env.process(ship_arrival(env,'Vessel_0'+str(i), port,g))
    yield env.timeout(0.00001)
   
    while True:
        yield env.timeout(np.random.exponential(217.18)+0.87359)
        i += 1
        yield env.process(ship_arrival(env, 'Vessel_0'+str(i), port,g))
        yield env.timeout(0.00001)

def progress_update(env,port,g):
    while True:
        print('Replication ', port.rep+1, ' out of ', g.REPLICATIONS, ' at time ', env.now, ': ', round((env.now/g.SIM_TIME)*100,2), '%')
        yield env.timeout(100)

def simulation(r,g):
    env = simpy.Environment()
    port = Port(env)
    port.rep = r
    port.filename = g.filename

    """
    Set up data frames for terminal and crane operations. Create a work list, holding container moves
    for each crane zone. Create a list of crane objects
    """
    for i in range(TOT_SPACE//2):
        port.WORK.append([])

    port.TERMINAL['Terminal'][:T1_SIZE] = 1
    port.TERMINAL['Terminal'][T1_SIZE:] = 2
    
    port.CRANES['Priority'] = port.CRANES['Priority'].apply(np.float)

    port.CRANES['Terminal'][:term_dic.get(1)[1]] = 1
    port.CRANES['Terminal'][term_dic.get(2)[0]:] = 2
    
    counter = 1
    
    for i in range(8): #Single lift cranes at terminal 1
        port.Crane_List.append(v.Crane(counter,1,i,1))
        port.CRANES.iloc[i]['Cranes']=counter
        counter += 1
    
    for i in range(8,14): #Dual lift cranes at terminal 1
        port.Crane_List.append(v.Crane(counter,2,i,1))
        port.CRANES.iloc[i]['Cranes']=counter
        counter += 1
    
    port.Crane_List.append(v.Crane(counter,1,T1_SIZE//2,2)) #First single lift crane at terminal 2
    port.CRANES.iloc[T1_SIZE//2]['Cranes']=counter
    counter += 1

    for i in range(1,12): #Dual lift cranes at terminal 2
        port.Crane_List.append(v.Crane(counter,2,i+T1_SIZE//2,2))
        port.CRANES.iloc[i+T1_SIZE//2]['Cranes']=counter
        counter += 1
    
    
    for i in range(12,16): #Remaining single lift cranes at terminal 2
        port.Crane_List.append(v.Crane(counter,1,i+T1_SIZE//2,2))
        port.CRANES.iloc[i+T1_SIZE//2]['Cranes']=counter
        counter += 1
    
    
    for i in range(0,T1_SIZE):
        port.TERMINAL.iloc[i]['Space'] = T1_SIZE-i
    
    for i in range(0,T2_SIZE):
        port.TERMINAL.iloc[i+T1_SIZE]['Space'] = T2_SIZE-i
    
    
    env.process(setup(env,port,g))
    env.process(intermittent_reassign(env,port))
    env.process(progress_update(env,port,g))
    env.run(g.SIM_TIME)
"""    
    print('Berth Queue:')
    for i in range(len(port.BERTH_QUEUE)):
        print(port.BERTH_QUEUE[i].name, ', ', port.BERTH_QUEUE[i].holds)
    print('\n')
    print('In process:')
    for i in range(len(port.IN_PROCESS)):
        print(port.IN_PROCESS[i].name, ', ', port.IN_PROCESS[i].holds, ' ', port.IN_PROCESS[i].remaining_workload, ' ', port.IN_PROCESS[i].total_workload, ' ', port.IN_PROCESS[i].t_berthing) 
    print('\n')
    
    print(len(port.COMPLETED))
    print('Completed:')    
    for i in range(len(port.COMPLETED)):
        print(port.COMPLETED[i].name, ', ', port.COMPLETED[i].t_arrival, ', ', port.COMPLETED[i].t_berthing, ', ', port.COMPLETED[i].t_start, ', ', port.COMPLETED[i].t_finish, ', ', port.COMPLETED[i].t_exit, ', ', port.COMPLETED[i].holds, ' ', port.COMPLETED[i].start_space, port.COMPLETED[i].total_workload)
    
    print(port.CRANES)
"""
if __name__ == '__main__':
    multiprocessing.freeze_support()

    g = SimVars()
    run_GUI(g)
    if g.RUN:
        print('filename = ', g.filename, ', Reps = ', g.REPLICATIONS, ', SimTime = ',
              g.SIM_TIME/(24*60), 'days, Berth priority = ', g.berth_discipline, ', using ',
              g.cores, ' cores')
        with open(g.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rep', 'Name', 'Type', 'No.holds', 'Terminal', 'Total moves', 'Total 20',
                             'Total 40', 'ATA', 'ATB', 'ATO', 'ATC', 'ATD'])
        Parallel(n_jobs=g.cores)(delayed(simulation)(K, g) for K in range(g.REPLICATIONS))
        print('\nFINISHED!')
    else:
        print('RUN CANCELED!')

   

