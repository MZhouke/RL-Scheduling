import numpy as np
import random
 
class Encode:
    def __init__(self,Matrix,Pop_size,J,J_num,M_num):
        self.Matrix=Matrix      #The operation time when the specific operation in job is done by a specific machine
        self.GS_num=int(0.6*Pop_size)      #Probability of global selection
        self.LS_num=int(0.2*Pop_size)     #Probability of local selection
        self.RS_num=int(0.2*Pop_size)     #Probability of random selection
        self.J=J                #represent how many operations in a job
        self.J_num=J_num        #the number of jobs
        self.M_num=M_num        #the number of machines
        self.CHS=[]
        self.Len_Chromo=0
        for i in J.values():
            self.Len_Chromo+=i
 
    #Prepare for generationg the operations
    def OS_List(self):
        OS_list=[]
        for k,v in self.J.items():
            OS_add=[k-1 for j in range(v)]
            OS_list.extend(OS_add)
        return OS_list
 
    #Generating initialized matrix
    def CHS_Matrix(self, C_num):  # C_num:the number of column needed
        return np.zeros([C_num, self.Len_Chromo], dtype=int)
 
    def Site(self,Job,Operation):
        O_num = 0
        for i in range(len(self.J)):
            if i == Job:
                return O_num + Operation
            else:
                O_num = O_num + self.J[i + 1]
        return O_num
 
    #initialization by global selection 
    def Global_initial(self):
        """
        When generate the machine decision part(MS) for each operation,
        regard the current load of machine + the time needed by the machine for this operation as a prior
        """
        MS=self.CHS_Matrix(self.GS_num)
        OS_list= self.OS_List()
        OS=self.CHS_Matrix(self.GS_num)
        for i in range(self.GS_num):
            Machine_time = np.zeros(self.M_num, dtype=float)  # initialize the machine time
            random.shuffle(OS_list)  # sort the generated operations
            OS[i] = np.array(OS_list)
            GJ_list = [i_1 for i_1 in range(self.J_num)]
            random.shuffle(GJ_list)
            for g in GJ_list:  # Randomly select the first job in the job set delete it from the job set
                h = self.Matrix[g]  # the operations included in the first job
                for j in range(len(h)):  # Begin to choose machine for the first operation in the job
                    D = h[j]
                    List_Machine_weizhi = []
                    for k in range(len(D)):  # The available machine for the operation and the time needed
                        Useing_Machine = D[k]
                        if Useing_Machine != 9999:  # determine the machine that can do this operation
                            List_Machine_weizhi.append(k)
                    Machine_Select = []
                    for Machine_add in List_Machine_weizhi:  
                        # Add the time needed by this machine for the operation into the machine time cumulated before
                        #  Select out the smallest among them
                        Machine_Select.append(Machine_time[Machine_add] + D[Machine_add])
                    Min_time = min(Machine_Select)
                    K = Machine_Select.index(Min_time)
                    I = List_Machine_weizhi[K]
                    Machine_time[I] += Min_time
                    site=self.Site(g,j)
                    MS[i][site] = K
        CHS1 = np.hstack((MS, OS))
        return CHS1
 
 
    #Initialized by local selection
    def Local_initial(self):
        """
        When generate the machine decision part(MS) for each operation,
        regard the the time needed by the machine for this operation as a prior
        """
        MS = self.CHS_Matrix(self.LS_num)
        OS_list = self.OS_List()
        OS = self.CHS_Matrix(self.LS_num)
        for i in range(self.LS_num):
            random.shuffle(OS_list)  # sort the generated operations
            OS_gongxu = OS_list
            OS[i] = np.array(OS_gongxu)
            GJ_list = [i_1 for i_1 in range(self.J_num)]
            for g in GJ_list:
                Machine_time = np.zeros(self.M_num)  # initialize the machine 
                h =self.Matrix[g]   # The first job and the time of its operations
                for j in range(len(h)):  # select machine for the operations in the first job
                    D = h[j]
                    List_Machine_weizhi = []
                    for k in range(len(D)):  # The available machine for the operation and the time needed
                        Useing_Machine = D[k]
                        if Useing_Machine == 9999:  # determine the machine that can do this operation
                            continue
                        else:
                            List_Machine_weizhi.append(k)
                    Machine_Select = []
                    for Machine_add in List_Machine_weizhi: 
                        # Add the time needed by this machine for the operation into the machine time cumulated before
                        # Compare and Select out the smallest among them
                        Machine_time[Machine_add] = Machine_time[Machine_add] + D[
                            Machine_add] 
                        Machine_Select.append(Machine_time[Machine_add])
                    Machine_Index_add = Machine_Select.index(min(Machine_Select))
                    site = self.Site(g, j)
                    MS[i][site] = MS[i][site] + Machine_Index_add
        CHS1 = np.hstack((MS, OS))
        return CHS1
 
    def Random_initial(self):
        """
        When generate the machine decision part(MS) for each operation,
        CHoose randomly
        """
        MS = self.CHS_Matrix(self.RS_num)
        OS_list = self.OS_List()
        OS = self.CHS_Matrix(self.RS_num)
        for i in range(self.RS_num):
            random.shuffle(OS_list)  # # sort the generated operations
            OS_gongxu = OS_list
            OS[i] = np.array(OS_gongxu)
            GJ_list = [i_1 for i_1 in range(self.J_num)]
            A = 0
            for gon in GJ_list:
                Machine_time = np.zeros(self.M_num)  # initialize the machine time
                g = gon  # Randomly select the first job in the job set delete it from the job set
                h = np.array(self.Matrix[g])  # The first job and the time of its operations
                for j in range(len(h)):  # select machine for the operations in the first job
                    D = np.array(h[j])
                    List_Machine_weizhi = []
                    Site=0
                    for k in range(len(D)):  # The available machine for the operation and the time needed
                        if D[k] == 9999:  # determine the machine that can do this operation
                            continue
                        else:
                            List_Machine_weizhi.append(Site)
                            Site+=1
                    Machine_Index_add = random.choice(List_Machine_weizhi)
                    MS[i][A] = MS[i][A] + Machine_Index_add
                    A += 1
        CHS1 = np.hstack((MS, OS))
        return CHS1