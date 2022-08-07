import matplotlib.pyplot as plt
from Jobs import Job
from Machines import Machine_Time_window
import numpy as np
 
class Decode:
    def __init__(self,J,Processing_time,M_num):
        self.Processing_time = Processing_time
        self.Scheduled = []  # Sorted operations
        self.M_num = M_num
        self.Machines = []  # Store information of machines
        self.fitness = 0
        self.J=J            #
        for j in range(M_num):
            self.Machines.append(Machine_Time_window(j))
        self.Machine_State = np.zeros(M_num, dtype=int)  #Which job is currently doing by the machine
        self.Jobs = []     #Store the information of jobs
        for k, v in J.items():
            self.Jobs.append(Job(k, v))
    #The timeline matrix and matrix indicate the order of machine.
    def Order_Matrix(self,MS):
        JM=[]
        T=[]
        Ms_decompose=[]
        Site=0
        for S_i in self.J.values():
            Ms_decompose.append(MS[Site:Site+S_i])
            Site+=S_i
        for i in range(len(Ms_decompose)):
            JM_i=[]
            T_i=[]
            for j in range(len(Ms_decompose[i])):
                O_j=self.Processing_time[i][j]
                M_ij=[]
                T_ij=[]
                for Mac_num in range(len(O_j)):  # Find out the machine time and machine order that MS represented
                    if O_j[Mac_num] != 9999:
                        M_ij.append(Mac_num)
                        T_ij.append(O_j[Mac_num])
                    else:
                        continue
                JM_i.append(M_ij[Ms_decompose[i][j]])
                T_i.append(T_ij[Ms_decompose[i][j]])
            JM.append(JM_i)
            T.append(T_i)
        return JM,T
    def Earliest_Start(self,Job,O_num,Machine):
        P_t=self.Processing_time[Job][O_num][Machine]
        last_O_end = self.Jobs[Job].Last_Processing_end_time  # The end time of the last operation
        Selected_Machine=Machine
        M_window = self.Machines[Selected_Machine].Empty_time_window()
        M_Tstart = M_window[0]
        M_Tend = M_window[1]
        M_Tlen = M_window[2]
        Machine_end_time = self.Machines[Selected_Machine].End_time
        ealiest_start = max(last_O_end, Machine_end_time)
        if M_Tlen is not None: 
            for le_i in range(len(M_Tlen)):
                if M_Tlen[le_i] >= P_t:
                    if M_Tstart[le_i] >= last_O_end:
                        ealiest_start=M_Tstart[le_i]
                        break
                    if M_Tstart[le_i] < last_O_end and M_Tend[le_i] - last_O_end >= P_t:
                        ealiest_start = last_O_end
                        break
        M_Ealiest = ealiest_start
        End_work_time = M_Ealiest + P_t
        return M_Ealiest, Selected_Machine, P_t, O_num,last_O_end,End_work_time
    #Decode information in MS,OS
    def Decode_1(self,CHS,Len_Chromo):
        MS=list(CHS[0:Len_Chromo])
        OS=list(CHS[Len_Chromo:2*Len_Chromo])
        Needed_Matrix=self.Order_Matrix(MS)
        JM=Needed_Matrix[0]
        for i in OS:
            Job=i
            O_num=self.Jobs[Job].Current_Processed()
            Machine=JM[Job][O_num]
            Para=self.Earliest_Start(Job,O_num,Machine)
            self.Jobs[Job]._Input(Para[0],Para[5],Para[1])
            if Para[5]>self.fitness:
                self.fitness=Para[5]
            self.Machines[Machine]._Input(Job,Para[0],Para[2],Para[3])
        return self.fitness
 
    def Gantt(self,Machines):
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'palegoldenrod', 'purple', 'pink', 'Thistle', 'Magenta',
             'SlateBlue', 'RoyalBlue', 'Cyan', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
             'navajowhite',
             'navy', 'sandybrown', 'moccasin']
        for i in range(len(Machines)):
            Machine=Machines[i]
            Start_time=Machine.O_start
            End_time=Machine.O_end
            for i_1 in range(len(End_time)):
                # plt.barh(i,width=End_time[i_1]-Start_time[i_1],height=0.8,left=Start_time[i_1],\
                #          color=M[Machine.assigned_task[i_1][0]],edgecolor='black')
                # plt.text(x=Start_time[i_1]+0.1,y=i,s=Machine.assigned_task[i_1])
                plt.barh(i, width=End_time[i_1] - Start_time[i_1], height=0.8, left=Start_time[i_1], \
                         color='white', edgecolor='black')
                plt.text(x=Start_time[i_1] + (End_time[i_1] - Start_time[i_1])/2-0.5, y=i, s=Machine.assigned_task[i_1][0])
        plt.yticks(np.arange(i + 1), np.arange(1, i + 2))
        plt.title('Scheduling Gantt chart')
        plt.ylabel('Machines')
        plt.xlabel('Time(s)')
        plt.show()