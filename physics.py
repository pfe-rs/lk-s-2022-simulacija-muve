import numpy as np
from math import sin,cos,pi,sqrt
from time import sleep

import vpython as vp

g = np.array([0.0,0.0,-9.81])
dt = 0.001
ro = 0.1225
Cd = 1.28

#region 3d transform funkcije
def translationMat(delta):
    return np.array([[1.0,0.0,0.0,delta[0]],
                     [0.0,1.0,0.0,delta[1]],
                     [0.0,0.0,1.0,delta[2]],
                     [0.0,0.0,0.0,1.0]])

def rotMatX(angle):
    return np.array([[1.0,0.0,0.0,0.0],
                     [0.0,cos(angle),sin(angle),0.0],
                     [0.0,-sin(angle),cos(angle),0.0],
                     [0.0,0.0,0.0,1.0]])

def rotMatY(angle):
    return np.array([[cos(angle),0.0,-sin(angle),0.0],
                     [0.0,1.0,0.0,0.0],
                     [sin(angle),0.0,cos(angle),0.0],
                     [0.0,0.0,0.0,1.0]])

def rotMatZ(angle):
    return np.array([[cos(angle),-sin(angle),0.0,0.0],
                     [sin(angle),cos(angle),0.0,0.0],
                     [0.0,0.0,1.0,0.0],
                     [0.0,0.0,0.0,1.0]])

def coordMat(delta,angles):
    mat = translationMat(delta)@rotMatZ(angles[2])@rotMatY(angles[1])@rotMatX(angles[0])
    return mat
    
def rotPivotX(point,pivot,angle):
    centered = point-pivot
    rotated = rotMatX(angle)@centered
    return rotated+pivot
    # print('X ',res)
    return res

def rotPivotY(point,pivot,angle):
    centered = point-pivot
    rotated = rotMatY(angle)@centered
    return rotated+pivot# print('Y ', res)
    return res

def rotPivotZ(point,pivot,angle):
    centered = point-pivot
    rotated = rotMatZ(angle)@centered
    return rotated+pivot# print('Z ',res)
    return res

def convertToGlobal(pos,rot,localCoords):
        return coordMat(pos,rot)@localCoords

def convertToLocal(pos,rot,globalCoords):
    return np.linalg.inv(coordMat(pos,rot))@globalCoords
#endregion

class Wing:
    def __init__(self,point1 = np.array([0.0,0.0,0.0]),point2 = np.array([0.0,0.0,0.0]),pivot = np.array([0.0,0.0,0.0])):
        self.pivot = pivot
        self.point1 = point1
        self.point2 = point2
        self.rotation = np.array([0.0,0.0,0.0])

    def getArea(self): # Vraca projekcije povrsine po x,y,z
        v1 = self.point1 - self.pivot
        v2 = self.point2 - self.pivot
        temp = np.absolute(np.cross(v1[:-1],v2[:-1]))
        temp = np.append(temp,1.0)
        return temp

    def getMiddle(self):
        return (self.point1+self.point2)/2

    def flapWing(self,angles):
        self.rotation += angles

        self.point1 = rotPivotX(self.point1,self.pivot,angles[0])
        self.point1 = rotPivotY(self.point1,self.pivot,angles[1])
        self.point1 = rotPivotZ(self.point1,self.pivot,angles[2])
        self.point2 = rotPivotX(self.point2,self.pivot,angles[0])
        self.point2 = rotPivotY(self.point2,self.pivot,angles[1])
        self.point2 = rotPivotZ(self.point2,self.pivot,angles[2])

    def flapWingConstrained(self,angles):

        maxVX = 0.06
        maxVY = 0.01
        maxVZ = 0.01
        if abs(angles[0]) > maxVX:
            angles[0] = maxVX * np.sign(angles[0])
        if abs(angles[1]) > maxVX:
            angles[1] = maxVY * np.sign(angles[1])
        if abs(angles[2]) > maxVX:
            angles[2] = maxVZ * np.sign(angles[2])
        
        limitX = 0.4
        limitY = pi/3
        limitZ = 0.4
        if self.rotation[0]+angles[0] > limitX or self.rotation[0]+angles[0] < -limitX:
            angles[0] = 0.0
        if self.rotation[1]+angles[1] > limitY or self.rotation[1]+angles[1] < -limitY:
            angles[1] = 0.0
        if self.rotation[2]+angles[2] > limitZ or self.rotation[2]+angles[2] < -limitZ:
            angles[2] = 0.0

        self.flapWing(angles)
        
class Fly:
    def __init__(self):
        self.position = np.array([0.0,0.0,0.0,1.0]) #Position i rotation sistema
        self.rotation = np.array([0.0,0.0,0.0])
        self.mass = 5

        self.blf = np.array([-1.0,2.0,0.0,1.0]) # Ove vrednosti su u lokalnom sistemu bube i one su konstantne
        self.brf = np.array([1.0,2.0,0.0,1.0])
        self.brb = np.array([1.0,-2.0,0.0,1.0])
        self.blb = np.array([-1.0,-2.0,0.0,1.0])
        self.tlf = np.array([-1.0,2.0,1.0,1.0])
        self.trf = np.array([1.0,2.0,1.0,1.0])
        self.trb = np.array([1.0,-2.0,1.0,1.0])
        self.tlb = np.array([-1.0,-2.0,1.0,1.0])


        self.lwing = Wing(np.array([-4.0,1.0,0.5,1.0]),np.array([-1.0,-1.0,0.5,1.0]),np.array([-1.0,1.0,0.5,1.0]))
        self.rwing = Wing(np.array([1.0,-1.0,0.5,1.0]),np.array([4.0,1.0,0.5,1.0]),np.array([1.0,1.0,0.5,1.0]))

class PhysicsEngine:
    def __init__(self):

        self.render3D = False

        self.f = Fly()

        self.v = np.array([0.0,0.0,0.0])
        self.w = np.array([0.0,0.0,0.0])

        # self.f.rotation = np.array([0.0,pi/4,0.0])
        # self.f.lwing.flapWing(np.array([pi/2,0.0,-0.0]))
        # self.f.rwing.flapWing(np.array([-pi/2,0.0,0.0]))
        
        self.midLLast = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.getMiddle())
        self.midRLast = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.getMiddle())


        if self.render3D:
            self.setup3D()
            self.pointerLx = vp.arrow()
            self.pointerLy = vp.arrow()
            self.pointerLz = vp.arrow()
            self.pointerRx = vp.arrow()
            self.pointerRy = vp.arrow()
            self.pointerRz = vp.arrow()

    def run(self,tMax,symetricWings):
        t = 0

        while(t < tMax):
            anglesLeft,anglesRight = superAwesomeMLFunkcija(self.f.position,self.f.rotation,self.v,self.w,self.f.lwing.rotation,self.f.rwing.rotation,symetricWings) # Nemam pojma kako bi se povezivalo ali pretpostavljam ovako nekako
            
            self.step(anglesLeft,anglesRight)

            if self.render3D:
                self.update3D()
                sleep(0.1)

            t+= dt


    def step(self,anglesLeft,anglesRight):
        Fl,Fr,pl,pr = self.calculateDrag()
        Fb = self.calculateBodyDrag()
        Q = self.f.mass * g
        pq = np.array([0.0,0.0,0.5,1.0])
        Ml = np.cross(Fl,convertToGlobal(np.array([0,0,0]),self.f.rotation,(pl-pq))[:-1])
        Mr = np.cross(Fr,convertToGlobal(np.array([0,0,0]),self.f.rotation,(pr-pq))[:-1])
        
        # print(Ml,Mr)
        M = Ml+Mr
        # M = 0
        acm = (Fl+Fr+Q+Fb) / self.f.mass
        Icm = np.array([1.0,1.0,1.0])
        alfa = M / Icm
        self.v += acm *dt
        self.w += alfa * dt
        self.w *= 0.8
        self.f.position[:-1] += self.v*dt
        self.f.rotation += self.w*dt
        self.f.lwing.flapWingConstrained(anglesLeft)
        self.f.rwing.flapWingConstrained(anglesRight)
        t+=dt

    def calculateBodyDrag(self):
        F = -1/2 * ro * Cd * 6 * self.v * np.absolute(self.v)
        # pointer = vp.arrow(pos=vp.vector(self.f.position[0],self.f.position[2],self.f.position[1]),axis=vp.vector(F[0]/10,F[2]/10,F[1]/10))
        return F

    def calculateDrag(self):
        l = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.getMiddle())
        r = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.getMiddle())
        dl = l - self.midLLast
        dr = r - self.midRLast

        self.midLLast = l
        self.midRLast = r

        al = convertToGlobal(np.array([0.0,0.0,0.0]),self.f.rotation,self.f.lwing.getArea())
        al = np.absolute(al)
        ar = convertToGlobal(np.array([0.0,0.0,0.0]),self.f.rotation,self.f.rwing.getArea())
        ar = np.absolute(ar)
        vl = dl[:-1]/dt
        vr = dr[:-1]/dt
        Fl = -1/2 * ro * Cd * al[:-1] * vl * vl * np.sign(vl)
        Fr = -1/2 * ro * Cd * ar[:-1] * vr * vr * np.sign(vr)
        # self.pointerLx.pos = vp.vec(l[0],l[2],l[1])
        # self.pointerLx.axis = vp.vec(vl[0],0.0,0.0)
        # self.pointerLy.pos = vp.vec(l[0],l[2],l[1])
        # self.pointerLy.axis = vp.vec(0.0,vl[2],0.0)
        # self.pointerLz.pos = vp.vec(l[0],l[2],l[1])
        # self.pointerLz.axis = vp.vec(0.0,0.0,vl[1])

        # self.pointerRx.pos = vp.vec(r[0],r[2],r[1])
        # self.pointerRx.axis = vp.vec(vr[0],0.0,0.0)
        # self.pointerRy.pos = vp.vec(r[0],r[2],r[1])
        # self.pointerRy.axis = vp.vec(0.0,vr[2],0.0)
        # self.pointerRz.pos = vp.vec(r[0],r[2],r[1])
        # self.pointerRz.axis = vp.vec(0.0,0.0,vr[1])
        return Fl,Fr,self.f.lwing.getMiddle(),self.f.rwing.getMiddle() # Sile su vektor duzine 3, l i r duzine 4 (dimenzija 4 je uvek 1 zbog homogenous transformations)
    
    def setup3D(self):
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.pivot)
        self.lp = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.point1)
        self.lp1 = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.point2)
        self.lp2 = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        self.lw = vp.triangle(vs=[self.lp,self.lp1,self.lp2])

        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.pivot)
        self.rp = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.point1)
        self.rp1 = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.point2)
        self.rp2 = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        self.rw = vp.triangle(vs=[self.rp,self.rp1,self.rp2])

        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.blb)
        self.blb = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.blf)
        self.blf = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.brb)
        self.brb = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.brf)
        self.brf = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        self.bodyb = vp.quad(vs=[self.blb,self.blf,self.brf,self.brb])

        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.tlb)
        self.tlb = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.tlf)
        self.tlf = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.trb)
        self.trb = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.trf)
        self.trf = vp.vertex(pos=vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1]))
        self.bodyt = vp.quad(vs=[self.tlb,self.tlf,self.trf,self.trb])

        self.bodyl = vp.quad(vs=[self.blb,self.blf,self.tlf,self.tlb])
        self.bodyr = vp.quad(vs=[self.brb,self.brf,self.trf,self.trb])
        self.bodyf = vp.quad(vs=[self.blf,self.brf,self.trf,self.tlf])
        self.bodyf = vp.quad(vs=[self.blb,self.brb,self.trb,self.tlb])

    def update3D(self):
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.pivot)
        self.lp.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.point1)
        self.lp1.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.lwing.point2)
        self.lp2.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])

        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.pivot)
        self.rp.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.point1)
        self.rp1.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.rwing.point2)
        self.rp2.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])

        self.lw = vp.triangle(vs=[self.lp,self.lp1,self.lp2])
        self.rw = vp.triangle(vs=[self.rp,self.rp1,self.rp2])
        
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.blb)
        self.blb.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.blf)
        self.blf.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.brb)
        self.brb.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.brf)
        self.brf.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])

        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.tlb)
        self.tlb.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.tlf)
        self.tlf.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.trb)
        self.trb.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])
        pomGlobal = convertToGlobal(self.f.position,self.f.rotation,self.f.trf)
        self.trf.pos = vp.vec(pomGlobal[0],pomGlobal[2],pomGlobal[1])

        self.bodyb = vp.quad(vs=[self.blb,self.blf,self.brf,self.brb])
        self.bodyt = vp.quad(vs=[self.tlb,self.tlf,self.trf,self.trb])

        self.bodyl = vp.quad(vs=[self.blb,self.blf,self.tlf,self.tlb])
        self.bodyr = vp.quad(vs=[self.brb,self.brf,self.trf,self.trb])
if __name__ == "__main__":
    p = PhysicsEngine()
    p.run(tMax=20,symetricWings=False)