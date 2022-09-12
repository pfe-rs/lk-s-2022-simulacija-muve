import matplotlib.pyplot as plt

import vpython as vp
import numpy as np
from math import sin,cos,pi

from time import sleep

g = np.array([0,0,-100.0])
dt = 0.001
ro = 1.2754
Cd = 0.5 # Ova vrednost je ukleta, ne menjati!

class Wing:
    def __init__(self,point1 = np.array([0.0,0.0,0.0]),point2 = np.array([0.0,0.0,0.0]),pivot = np.array([0.0,0.0,0.0])):
        self.pivot = pivot
        self.point1 = point1
        self.point2 = point2
 
    def rotateWing(self,x,y,z): # Pomeranje krila oko pivota u radijanima
        self.point1 = rotateAroundPivotX(self.pivot,self.point1,x)
        self.point1 = rotateAroundPivotY(self.pivot,self.point1,y)
        self.point1 = rotateAroundPivotZ(self.pivot,self.point1,z)

        self.point2 = rotateAroundPivotX(self.pivot,self.point2,x)
        self.point2 = rotateAroundPivotY(self.pivot,self.point2,y)
        self.point2 = rotateAroundPivotZ(self.pivot,self.point2,z)

    def getArea(self): # Vraca projekcije povrsine po x,y,z
        v1 = self.point1 - self.pivot
        v2 = self.point2 - self.pivot
        return np.absolute(np.cross(v1,v2))

    def getMiddle(self):
        return (self.point1+self.point2)/2

class Fly:
    def __init__(self):
        self.blf = np.array([-1.0,2.0,0.0])
        self.brf = np.array([1.0,2.0,0.0])
        self.brb = np.array([1.0,-2.0,0.0])
        self.blb = np.array([-1.0,-2.0,0.0])
        self.tlf = np.array([-1.0,2.0,1.0])
        self.trf = np.array([1.0,2.0,1.0])
        self.trb = np.array([1.0,-2.0,1.0])
        self.tlb = np.array([-1.0,-2.0,1.0])
        self.lwing = Wing(np.array([-4.0,1.0,0.0]),np.array([-1.0,-1.0,0.0]),np.array([-1.0,1.0,0.0]))
        self.rwing = Wing(np.array([1.0,-1.0,0.0]),np.array([4.0,1.0,0.0]),np.array([1.0,1.0,0.0]))
        self.mass = 5

    def getMidpoint(self):
        return (self.tlf+self.brb)/2
    
    def moveFly(self,delta):
        self.blf += delta
        self.brf += delta
        self.brb += delta
        self.blb += delta
        self.tlf += delta
        self.trf += delta
        self.trb += delta
        self.tlb += delta

        self.lwing.pivot += delta
        self.lwing.point1 += delta
        self.lwing.point2 += delta

        self.rwing.pivot += delta
        self.rwing.point1 += delta
        self.rwing.point2 += delta

    def rotateFly(self,pivot,x,y,z): # Sramota me je ove funkcije
        self.blf = rotateAroundPivotX(pivot,self.blf,x)
        self.blf = rotateAroundPivotY(pivot,self.blf,y)
        self.blf = rotateAroundPivotZ(pivot,self.blf,z)  

        self.brf = rotateAroundPivotX(pivot,self.brf,x)
        self.brf = rotateAroundPivotY(pivot,self.brf,y)
        self.brf = rotateAroundPivotZ(pivot,self.brf,z)

        self.brb = rotateAroundPivotX(pivot,self.brb,x)
        self.brb = rotateAroundPivotY(pivot,self.brb,y)
        self.brb = rotateAroundPivotZ(pivot,self.brb,z)

        self.blb = rotateAroundPivotX(pivot,self.blb,x)
        self.blb = rotateAroundPivotY(pivot,self.blb,y)
        self.blb = rotateAroundPivotZ(pivot,self.blb,z)


        self.tlf = rotateAroundPivotX(pivot,self.tlf,x)
        self.tlf = rotateAroundPivotY(pivot,self.tlf,y)
        self.tlf = rotateAroundPivotZ(pivot,self.tlf,z)

        self.trf = rotateAroundPivotX(pivot,self.trf,x)
        self.trf = rotateAroundPivotY(pivot,self.trf,y)
        self.trf = rotateAroundPivotZ(pivot,self.trf,z)

        self.trb = rotateAroundPivotX(pivot,self.trb,x)
        self.trb = rotateAroundPivotY(pivot,self.trb,y)
        self.trb = rotateAroundPivotZ(pivot,self.trb,z)

        self.tlb = rotateAroundPivotX(pivot,self.tlb,x)
        self.tlb = rotateAroundPivotY(pivot,self.tlb,y)
        self.tlb = rotateAroundPivotZ(pivot,self.tlb,z)

        self.lwing.point1 = rotateAroundPivotX(pivot,self.lwing.point1,x)
        self.lwing.point1 = rotateAroundPivotY(pivot,self.lwing.point1,y)
        self.lwing.point1 = rotateAroundPivotZ(pivot,self.lwing.point1,z)
        self.lwing.point2 = rotateAroundPivotX(pivot,self.lwing.point2,x)
        self.lwing.point2 = rotateAroundPivotY(pivot,self.lwing.point2,y)
        self.lwing.point2 = rotateAroundPivotZ(pivot,self.lwing.point2,z)
        self.lwing.pivot = rotateAroundPivotX(pivot,self.lwing.pivot,x)
        self.lwing.pivot = rotateAroundPivotY(pivot,self.lwing.pivot,y)
        self.lwing.pivot = rotateAroundPivotZ(pivot,self.lwing.pivot,z)

        self.rwing.point1 = rotateAroundPivotX(pivot,self.rwing.point1,x)
        self.rwing.point1 = rotateAroundPivotY(pivot,self.rwing.point1,y)
        self.rwing.point1 = rotateAroundPivotZ(pivot,self.rwing.point1,z)
        self.rwing.point2 = rotateAroundPivotX(pivot,self.rwing.point2,x)
        self.rwing.point2 = rotateAroundPivotY(pivot,self.rwing.point2,y)
        self.rwing.point2 = rotateAroundPivotZ(pivot,self.rwing.point2,z)
        self.rwing.pivot = rotateAroundPivotX(pivot,self.rwing.pivot,x)
        self.rwing.pivot = rotateAroundPivotY(pivot,self.rwing.pivot,y)
        self.rwing.pivot = rotateAroundPivotZ(pivot,self.rwing.pivot,z)


def rotateAroundPivotX(pivot,point,angle):
    centered = point-pivot
    rotated = np.matmul([[1,0,0],[0, cos(angle),-sin(angle)],[0,sin(angle),cos(angle)]], centered)
    return rotated + pivot
def rotateAroundPivotY(pivot,point,angle):
    centered = point-pivot
    rotated = np.matmul([[cos(angle),0,sin(angle)],[0,1,0],[-sin(angle),0,cos(angle)]], centered)
    return rotated + pivot
def rotateAroundPivotZ(pivot,point,angle):
    centered = point-pivot
    rotated = np.matmul([[cos(angle),-sin(angle),0],[sin(angle),cos(angle),0],[0,0,1]], centered)
    return rotated + pivot

def CalculateDrag(fly,lLast,rLast):
    l = fly.lwing.getMiddle()
    r = fly.rwing.getMiddle()
    dl = l - lLast
    dr = r - rLast
    al = fly.lwing.getArea()
    ar = fly.rwing.getArea()
    vl = dl/dt
    vr = dr/dt
    Fl = -1/2 * ro * Cd * al * vl * np.absolute(vl)
    Fr = -1/2 * ro * Cd * ar * vr * np.absolute(vl)
    # print(al,dl)
    return Fl,Fr,l,r

class PhysicsSim:
    def __init__(self):
        self.f = Fly()
        

        self.v = np.array([0.0,0.0,0.0])
        self.w = np.array([0.0,0.0,0.0])

        # self.f.lwing.rotateWing(0,pi/2,0)
        # self.f.rwing.rotateWing(0,-pi/2,0)

        self.midLLast = self.f.lwing.getMiddle()
        self.midRLast = self.f.rwing.getMiddle()

        self.setup3D()

    def setup3D(self):
        self.lp = vp.vertex(pos=vp.vec(self.f.lwing.pivot[0],self.f.lwing.pivot[2],self.f.lwing.pivot[1]))
        self.lp1 = vp.vertex(pos=vp.vec(self.f.lwing.point1[0],self.f.lwing.point1[2],self.f.lwing.point1[1]))
        self.lp2 = vp.vertex(pos=vp.vec(self.f.lwing.point2[0],self.f.lwing.point2[2],self.f.lwing.point2[1]))
        self.lw = vp.triangle(vs=[self.lp,self.lp1,self.lp2])
        self.rp = vp.vertex(pos=vp.vec(self.f.rwing.pivot[0],self.f.rwing.pivot[2],self.f.rwing.pivot[1]))
        self.rp1 = vp.vertex(pos=vp.vec(self.f.rwing.point1[0],self.f.rwing.point1[2],self.f.rwing.point1[1]))
        self.rp2 = vp.vertex(pos=vp.vec(self.f.rwing.point2[0],self.f.rwing.point2[2],self.f.rwing.point2[1]))
        self.rw = vp.triangle(vs=[self.rp,self.rp1,self.rp2])

        self.v1 = vp.vertex(pos=vp.vec(self.f.blf[0],self.f.blf[2],self.f.blf[1]))
        self.v2 = vp.vertex(pos=vp.vec(self.f.blb[0],self.f.blb[2],self.f.blb[1]))
        self.v3 = vp.vertex(pos=vp.vec(self.f.brb[0],self.f.brb[2],self.f.brb[1]))
        self.v4 = vp.vertex(pos=vp.vec(self.f.brf[0],self.f.brf[2],self.f.brf[1]))        

    def update3D(self):
        self.lp.pos = vp.vec(self.f.lwing.pivot[0],self.f.lwing.pivot[2],self.f.lwing.pivot[1])
        self.lp1.pos = vp.vec(self.f.lwing.point1[0],self.f.lwing.point1[2],self.f.lwing.point1[1])
        self.lp2.pos = vp.vec(self.f.lwing.point2[0],self.f.lwing.point2[2],self.f.lwing.point2[1])

        self.rp.pos = vp.vec(self.f.rwing.pivot[0],self.f.rwing.pivot[2],self.f.rwing.pivot[1])
        self.rp1.pos = vp.vec(self.f.rwing.point1[0],self.f.rwing.point1[2],self.f.rwing.point1[1])
        self.rp2.pos = vp.vec(self.f.rwing.point2[0],self.f.rwing.point2[2],self.f.rwing.point2[1])

        self.lw = vp.triangle(vs=[self.lp,self.lp1,self.lp2])
        self.rw = vp.triangle(vs=[self.rp,self.rp1,self.rp2])

        self.v1.pos = vp.vec(self.f.blf[0],self.f.blf[2],self.f.blf[1])
        self.v2.pos = vp.vec(self.f.blb[0],self.f.blb[2],self.f.blb[1])
        self.v3.pos = vp.vec(self.f.brb[0],self.f.brb[2],self.f.brb[1])
        self.v4.pos = vp.vec(self.f.brf[0],self.f.brf[2],self.f.brf[1])

        self.body = vp.quad(vs=[self.v1,self.v2,self.v3,self.v4])
    
    def plot3d(self):
        fig = plt.figure(figsize=(4,4))

        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)

        ax.scatter(self.f.blf[0],self.f.blf[1],self.f.blf[2],color = 'red')
        ax.scatter(self.f.brf[0],self.f.brf[1],self.f.brf[2],color = 'red')
        ax.scatter(self.f.brb[0],self.f.brb[1],self.f.brb[2],color = 'red')
        ax.scatter(self.f.tlf[0],self.f.tlf[1],self.f.tlf[2],color = 'red')

        ax.scatter(self.f.lwing.pivot[0],self.f.lwing.pivot[1],self.f.lwing.pivot[2],color = 'blue')

        ax.scatter(self.f.rwing.pivot[0],self.f.rwing.pivot[1],self.f.rwing.pivot[2],color = 'green')

        ax.scatter(self.f.lwing.point1[0],self.f.lwing.point1[1],self.f.lwing.point1[2],color = 'blue')
        ax.scatter(self.f.lwing.point2[0],self.f.lwing.point2[1],self.f.lwing.point2[2],color = 'blue')
        

        ax.scatter(self.f.rwing.point1[0],self.f.rwing.point1[1],self.f.rwing.point1[2],color = 'green')
        ax.scatter(self.f.rwing.point2[0],self.f.rwing.point2[1],self.f.rwing.point2[2],color = 'green')
        
        plt.show()

    def runSim(self):
        t = 0
        while(True):
            t+=dt
            Fl,Fr,pl,pr = CalculateDrag(self.f,self.midLLast,self.midRLast)
            Q = self.f.mass * g
            pq = self.f.getMidpoint() # Centar mase

            self.midLLast = self.f.lwing.getMiddle()
            self.midRLast = self.f.rwing.getMiddle()

            Ml = np.cross((pl-pq),Fl)
            Mr = np.cross((pr-pq),Fr)
            
            M = Ml+Mr
            acm = (Fl + Fr + Q) / self.f.mass

            Icm = np.array([1,1,1])
            
            # print(Fl,Fr,self.v)
            alfa = M / Icm
            self.v += acm * dt
            self.w += alfa * dt
                
            self.f.moveFly(self.v * dt)
            
            x = self.w[0]*dt
            y = self.w[1]*dt
            z = self.w[2]*dt
            self.f.rotateFly(pq,x,y,z)
            self.update3D()
            self.f.lwing.rotateWing(0.05,0.0,0.0)
            self.f.rwing.rotateWing(0.05,0.0,0.0)
            sleep(0.1)

            
if __name__ == "__main__":
    p = PhysicsSim()
    p.runSim()