# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:53:24 2017

@author: Swopnil Ojha
"""
import numpy as np 
import math 
import matplotlib.pyplot as plt 

class foundation(object):    
    def __init__ (self,P,cordinates,b,D,E,ks,density=25,selfweight=True,spacing=True,meshsize=0.5):
        self.P = np.concatenate(np.array(P))
        if spacing:
            elements = cordinates
            sa = np.ndarray(len(cordinates)) 
            sa[0] = cordinates[0]
            for i in range(1,len(cordinates)):
                sa[i] = sa[i-1] + cordinates[i]
        else:
            sa = cordinates
            elements = np.ndarray(len(cordinates))
            elements[0] = cordinates[0]
            for i in range(1, len(cordinates)):
                elements[i] = cordinates[i]-cordinates[i-1]                
        self.elements = elements            
        self.sa = sa           
        self.b = b
        self.D = D
        self.E = E
        self.ks = ks
        self.meshsize = meshsize
        self.density = density
        self.weight = selfweight
        self.elem_new,self.sa_new,self.P_new = self.arrangement()
        self.X,self.R,self.q = self.disp_react()
        
    def element_ASA(self,E,I,L):
        return np.array([[4*E*I/L,6*E*I/L**2,2*E*I/L,-6*E*I/L**2],
                         [6*E*I/L**2,12*E*I/L**3,6*E*I/L**2,-12*E*I/L**3],
                         [2*E*I/L,6*E*I/L**2,4*E*I/L,-6*E*I/L**2],
                         [-6*E*I/L**2,-12*E*I/L**3,-6*E*I/L**2,12*E*I/L**3]])
                         
    
    def arrangement(self):#generating the P and element matrix 
    # Create Mesh     
        elem = [values for values in self.elements]
        for i in range(len(elem)):
            if elem[i]>self.meshsize:
                piv = math.ceil(elem[i]/self.meshsize) 
                val = elem[i]/piv
                elem[i] = [val for i in range(piv)]
        
        ele = []
        for values in elem:
            if not isinstance(values,(list)):
                ele.append([values])
            else:
                ele.append(values)                
        elem = np.concatenate(np.array(ele))#element sizes        
        sa = np.ndarray(len(elem)) #element sizes from origin 
        sa[0] = elem[0]
        for i in range(1,len(elem)):
            sa[i] = sa[i-1] + elem[i]
        n_nodes = len(sa)+1  #number of nodes 
        P = np.zeros(2*n_nodes)   #total load matrix 
        p_space = [values for values in self.sa[:-1]]
        loc = [] #determine the location of loads
        for values in p_space:
            loc.append((np.where(np.logical_and(sa>values-0.01,sa<values+0.01))[0][0]+1)*2)
            loc.append((np.where(np.logical_and(sa>values-0.01,sa<values+0.01))[0][0]+1)*2+1)
        for i in range(0,len(self.P)):
           P[loc[i]] = self.P[i]
        if self.weight:
            for i in range(len(elem)):
                #P[2*i] += -self.density*self.b*self.D*elem[i]**2/12     
                #P[2*i+2] += self.density*self.b*self.D*elem[i]**2/12
                P[2*i+1] += self.density*self.b*self.D*elem[i]/2
                P[2*i+3] += self.density*self.b*self.D*elem[i]/2
        return elem,sa,P    
            
    def joint_springs(self):
        #elem = np.array([0.2,0.2,0.3,0.610,1.070,1.070,0.910,0.610,0.230,0.230,0.450,0.5])
        K = np.ndarray(len(self.elem_new)+1)
        K[0] = 2*self.elem_new[0]/2*self.b*self.ks
        K[-1] = 2*self.elem_new[-1]/2*self.b*self.ks
        for i in range(1,len(K)-1):
            K[i] = (self.elem_new[i-1]+self.elem_new[i])/2*self.b*self.ks
        return K
    
    def disp_react(self):
        #elem = np.array([0.2,0.2,0.3,0.610,1.070,1.070,0.910,0.610,0.230,0.230,0.450,0.5])
        #P = np.array([0.,0,-108,1350,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81.,2025,0,0,0,0,0,0])
        k = self.joint_springs()
        g_ASA = np.zeros((len(self.P_new),len(self.P_new)))
        a = np.ndarray(len(self.elem_new),dtype = int)
        b = np.ndarray(len(self.elem_new),dtype = int)
        I = self.b*self.D**3/12
        # element node numbers for global matrix formation 
        a[0] = 0
        b[0] = 4
        for i in range(1,len(a)):
            a[i] = a[i-1] + 2 
            b[i] = b[i-1] + 2 
        
        for i in range(len(self.elem_new)):
            g_ASA[a[i]:b[i],a[i]:b[i]] += self.element_ASA(self.E,I,self.elem_new[i])
        k_new = np.ndarray(len(self.P_new))
        
        # addition of node springs to the diagonal terms of the global matrix 
        k_new[1::2] = k        
        for i in range(1,len(self.P_new),2):
            g_ASA[i,i] += k_new[i]        
        X = np.linalg.solve(g_ASA,self.P_new)
        R = np.multiply(k,X[1::2])
        q = np.multiply(self.ks,X[1::2])
        return X,R,q
        
    def Element_ASA_node(self,element_number):
        #elem = np.array([0.2,0.2,0.3,0.610,1.070,1.070,0.910,0.610,0.230,0.230,0.450,0.5])
        I = self.b*self.D**3/12
        return self.element_ASA(self.E,I,self.elem_new[element_number])
        
    def Internal_forces(self,element_number):
        X_current = np.ndarray(4)
        for i in range(4):
            X_current[i] = self.X[element_number*2+i]
        return np.matmul(self.Element_ASA_node(element_number),X_current)
        
    def BMD_SFD(self):
        #elem = np.array([0.2,0.2,0.3,0.610,1.070,1.070,0.910,0.610,0.230,0.230,0.450,0.5])
        BMD = np.ndarray((len(self.elem_new)+1),dtype ='float32')
        BMD[0] = self.Internal_forces(0)[0]
        BMD[-1] = self.Internal_forces(len(self.elem_new)-1)[-2]
        for i in range(1,len(self.elem_new)):
            BMD[i] = self.Internal_forces(i)[0]
        SFD = np.ndarray((len(self.elem_new)+1)*2, dtype = 'float32')        
        for i in range(0,len(self.elem_new)):
            SFD[i*2+1] = -self.Internal_forces(i)[-1] 
            SFD[2*i] = -SFD[i*2-1]
        SFD[0] = 0
        SFD[-1] = 0.
        SFD[-2] = SFD[-3]        
        sa_BM = self.sa_new #element sizes from origin 
        sa_BM = np.append(np.array([0]),sa_BM)
        sa_SF = np.ndarray(len(SFD))  
        sa_SF[::2] = sa_BM
        sa_SF[1::2] = sa_BM
        return BMD,SFD,sa_BM,sa_SF 
            
    def plotting(self):
        yb,ys,xb,xs = self.BMD_SFD()
     
        #Bending moment diagram
        fig = plt.figure('Internal Force Diagram')
        ax1 = fig.add_subplot(2,1,1)       
        ax1.plot(xb,yb,linewidth = 1.5)
        
        if abs(max(yb)) >= abs(min(yb)):
            ybmax = abs(max(yb))
        else:
            ybmax = abs(min(yb))
        ax1.axis([0,sum(self.elem_new),-(ybmax+ybmax/20), (ybmax+ybmax/20)])
        yb1 = np.zeros(len(xb))
        ax1.plot(xb,yb1,'r',linewidth = .5)
        ax1.fill_between(xb,yb,yb1,facecolor = 'green', alpha =0.2)
        ax = plt.gca()
        ax.invert_yaxis()
        #ax1.set_xlabel('Distance(m)',fontsize = 13)
        ax1.set_ylabel('Bending Moment (kNm)', fontsize = 13)
        ax1.set_title('Bending Moment / Shear Force', fontsize = 15)
        #FollowDotCursor(ax1, x, yb, tolerance=20)
        plt.show()
        
        
a = foundation([[0,1350],[0,2025]],[0.2,5,1.18],b=2.65,D=0.6,E=21500000,ks=12000,selfweight=False,spacing=True)
a.plotting()

#elem,sa,P,loc = a.arrangement()            
            
            
            
            
