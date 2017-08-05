# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:54:46 2016

@author: Swopnil Ojha
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.spatial as spatial

def fmt(x, y):
    return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)

class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    http://stackoverflow.com/a/4674445/190597 (Joe Kington)
    http://stackoverflow.com/a/13306887/190597 (unutbu)
    http://stackoverflow.com/a/15454427/190597 (unutbu)
    """
    def __init__(self, ax, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        try:
            x = np.asarray(x, dtype='float')
        except (TypeError, ValueError):
            x = np.asarray(mdates.date2num(x), dtype='float')
        y = np.asarray(y, dtype='float')
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        self._points = np.column_stack((x, y))
        self.offsets = offsets
        y = y[np.abs(y-y.mean()) <= 3*y.std()]
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        plt.connect('motion_notify_event', self)

    def scaled(self, points):
        points = np.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(x, y))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._points[idx]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]



class combined_footing():
    
    def __init__(self,n,load,space):
        #n = number of loads, load = factored load, space = spacings        
        self.n = n
        self.load = load
        self.space = space
        try:
            isinstance(self.load,(list,tuple)) and \
                isinstance(self.space,(list,tuple))
        except:
            print('Enter the loading and spacing inside brackets')

    def spacing(self): #total length
        return sum(np.array(self.space))
        
    def spacing_from_origin(self):
        sa = np.ndarray(len(self.space)) 
        sa[0] = self.space[0]
        for i in range(1,len(self.space)):
            sa[i] = sa[i-1] + self.space[i]
        return sa
        
    def order(self): #order pair of spacings
        c = self.spacing_from_origin()
        c = np.insert(c,0,0)
        d = np.ndarray(len(c)-1)
        e = np.ndarray(len(c)-1)
        for i in range(len(c)-1):
            d[i] = c[i]
        for i in range(1,len(c)):
            e[i-1] = c[i]
        return d,e
        
    def sum_load(self): #sum of loads
        return sum(np.array(self.load))        
     
    def cg(self): #calculation of centre of gravity 
    # cg from left to right
        sum = 0
        sa = self.space[0]         
        for i in range(len(self.load)):
            sum += self.load[i]*sa
            sa += self.space[i+1]
        return sum/self.sum_load() 
        
    def eccentricity(self): #calculation of eccentricity 
        return -self.spacing()/2 + self.cg()
        
    def intensity(self): #intensity of bearing pressure 
        qmax = self.sum_load()*1/self.spacing()+\
            self.sum_load()*1*abs(self.eccentricity())/(self.spacing()**2/6)
        qmin = self.sum_load()*1/self.spacing()-\
            self.sum_load()*1*abs(self.eccentricity())/(self.spacing()**2/6)
        if qmax >0 and qmin >0: 
            x = np.linspace(0,self.spacing(),4*self.spacing())        
            q = np.ndarray(len(x))
            wb = np.ndarray(len(x))
            ws = np.ndarray(len(x))
            if self.eccentricity()<0:
                m = -(qmax-qmin)/self.spacing()
                c = qmax
                q = m*x+c
                wb = q*x**2/2+1/2*(c-q)*x*2/3*x
                ws = q*x+1/2*(c-q)*x
            elif self.eccentricity()==0:
                q = 0*x+qmin             
                wb = qmin*x**2/2
                ws = qmin*x
            else:
                m = (qmax-qmin)/self.spacing()
                c = qmin
                q = m*x+c
                wb = c*x**2/2+1/2*(q-c)*x*1/3*x
                ws = c*x+1/2*(q-c)*x
        else:
            statement = 'The footing fails in overturning'
            return statement    
        return x,q,wb,ws

        
    def internal_force(self): #internal force calculation 
        try:
            x,q,yb,ys = self.intensity()
            sa = np.array(self.spacing_from_origin())        
        # Calculation of bending moment and shear force
            for i in range(len(x)):
                for j in range(self.n):
                    if (x[i]-sa[j])>=0:
                        yb[i] = yb[i] - np.array(self.load)[j]*(x[i]-sa[j])
                        ys[i] = ys[i] - np.array(self.load)[j]
            return x,q,yb,ys
        except:
            print ('The footing fails in overturning')
            
    def contact_pressure(self):
        try:        
            x,_,_,_ = self.intensity()
            c_pressure = np.ndarray(len(x))
            c_pressure = self.sum_load()/(1.5*self.spacing()) + \
                self.sum_load()*self.eccentricity()/(1.5*self.spacing()**3/12)*\
                (x-self.spacing()/2)
                
            return c_pressure
        except:
            print ('')
                    
        
    def plotting(self): #plotting of bending moment, shear force and free body 
        try:       
            x,q,yb,ys = self.internal_force()
            #Bending moment diagram
            fig = plt.figure('Internal Force Diagram')
            ax1 = fig.add_subplot(2,1,1)       
            ax1.plot(x,yb,linewidth = 1.5)
            
            if abs(max(yb)) >= abs(min(yb)):
                ybmax = abs(max(yb))
            else:
                ybmax = abs(min(yb))
            ax1.axis([0,self.spacing(),-(ybmax+ybmax/20), (ybmax+ybmax/20)])
            yb1 = np.zeros(len(x))
            ax1.plot(x,yb1,'r',linewidth = .5)
            ax1.fill_between(x,yb,yb1,facecolor = 'green', alpha =0.2)
            ax = plt.gca()
            ax.invert_yaxis()
            #ax1.set_xlabel('Distance(m)',fontsize = 13)
            ax1.set_ylabel('Bending Moment (kNm)', fontsize = 13)
            ax1.set_title('Bending Moment / Shear Force', fontsize = 15)
            FollowDotCursor(ax1, x, yb, tolerance=20)
            
            # Shear force diagram 
            ax2 = fig.add_subplot(2,1,2)        
            ax2.plot(x,ys,'+-',linewidth = 1.5)
            
            if abs(max(ys)) > abs(min(ys)):
                ysmax = abs(max(ys))
            else:
                ysmax = abs(min(ys))
            ys1 = np.zeros(len(x))
            ax2.plot(x,ys1,'r',linewidth = .5)
            ax2.fill_between(x,ys,ys1,facecolor = 'yellow', alpha =0.2)        
            ax2.axis([0, self.spacing(), -(ysmax+ysmax/20), (ysmax+ysmax/20)])
            ax2.set_ylabel('Shear Force (kN)', fontsize = 13)
            ax2.set_xlabel('Distance (m)', fontsize = 13)        
            FollowDotCursor(ax2, x, ys, tolerance=20)        
            fig.tight_layout()
            
            #Loading Diagram
            fig2 = plt.figure('Loading Diagram')
            ax3 = fig2.add_subplot(111)
            ax3.plot(x,q,linewidth = 1.5)
            yl = np.zeros(len(x))
            ax3.plot(x,yl,'r',linewidth = 1)  
            ax3.axis([0, self.spacing(), -(max(self.load)+max(self.load)/2),\
                max(q)+max(q)])              
            ax3.fill_between(x,q,yl,facecolor = 'green', alpha =0.2)        
            
            #Drawing soil intensity arrow 
            for i in range(len(x)):        
                ax3.annotate("",(x[i],yl[i]),(x[i],q[i]),\
                    arrowprops = dict(arrowstyle='->'))     
            sa = self.spacing_from_origin()
            
            #Drawing the loading in the gravity direction        
            for i in range(self.n):
                ax3.annotate('',(sa[i],yl[i]),\
                    (sa[i],-self.load[i]),arrowprops = dict(arrowstyle= '->'))
                ax3.text(sa[i],-self.load[i],'{0:.2f} kN'.format(self.load[i]),
                    horizontalalignment = 'left')
            d,e = self.order()
            
            #writing the spacing between the loads        
            for i in range(len(self.space)):
                ax3.annotate('',(d[i],-max(self.load)/2),(e[i],-max(self.load)/2),\
                    arrowprops = dict(arrowstyle = '<->'))
                ax3.text((d[i]+e[i])/2,-max(self.load)/2,'{0:.2f}'.format(self.space[i]),\
                    horizontalalignment = 'center', verticalalignment = 'bottom')
            
            #writing the max, min pressure intensity,eccentricity and contact pressure 
            ax3.text(self.spacing()*.5,max(q)*1.75,\
                    'qmax = {0:.2f}kN/m \nqmin = {1:.2f}kN/m'.format(max(q),min(q)))
            ax3.text(self.spacing()*0.45,-max(self.load)*1.3,\
                    'eccentricity = {0:.2f}m\nmax contact pressure = {1:.2f}kN/m'\
                    .format(self.eccentricity(),max(abs(self.contact_pressure()))))
            ax = plt.gca()
            ax.invert_yaxis()
            ax3.set_xlabel('Distance(m)',fontsize = 13)
            ax3.set_ylabel('Load',fontsize = 13)
            ax3.set_title('FreeBody Diagram', fontsize = 15)
        except:
            print("")
        
a = combined_footing(2,[249*1.5,190*1.5],[0.6,1.0668,0.5]) 
b = combined_footing(2,[425*1.5,466*1.5],[1.2,1.850,1.2])
c = combined_footing(2,[273,403],[1,1.850,1])
a.plotting()


        
        