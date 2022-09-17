from physics import Wing,Fly,convertToGlobal
from bugControll import Bug

import vpython as vp
import numpy as np
from time import sleep
import pickle


class renderer3D:
    def __init__(self):
        self.savefile_name = "generation-10-bestBug4"

    def setup3DBest(self):
        self.bugs = []
        with open(self.savefile_name, "rb") as f:
            self.bugs = pickle.load(f)
        self.bestbug = self.bugs[0].bugParams
        self.f = Fly()

        framenum = len(self.bestbug["position"][0])
        print(framenum)

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


        xAxis = vp.curve(vp.vector(0,0,0), vp.vector(20,0,0))
        yAxis = vp.curve(vp.vector(0,0,0), vp.vector(0,0,20))
        zAxis = vp.curve(vp.vector(0,0,0), vp.vector(0,20,0))

        for i in range(framenum-1):
            self.update3DBest(i+1)

    def update3DBest(self,frame):
        xpos = self.bestbug["position"][0][frame]
        ypos = self.bestbug["position"][1][frame]
        zpos = self.bestbug["position"][2][frame]

        xrot = self.bestbug["rotation"][0][frame]
        yrot = self.bestbug["rotation"][1][frame]
        zrot = self.bestbug["rotation"][2][frame]

        self.f.position = np.array([xpos,ypos,zpos])
        self.f.rotation = np.array([xrot,yrot,zrot])

        dlx = self.bestbug["lwingRotation"][0][frame] - self.bestbug["lwingRotation"][0][frame-1]
        dly = self.bestbug["lwingRotation"][1][frame] - self.bestbug["lwingRotation"][1][frame-1]
        dlz = self.bestbug["lwingRotation"][2][frame] - self.bestbug["lwingRotation"][2][frame-1]

        drx = self.bestbug["rwingRotation"][0][frame] - self.bestbug["rwingRotation"][0][frame-1]
        dry = self.bestbug["rwingRotation"][1][frame] - self.bestbug["rwingRotation"][1][frame-1]
        drz = self.bestbug["rwingRotation"][2][frame] - self.bestbug["rwingRotation"][2][frame-1]

        self.f.lwing.flapWing(np.array([dlx,dly,dlz]))
        self.f.rwing.flapWing(np.array([drx,dry,drz]))

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

        sleep(0.01)

    def setup3DAll(self):
        return

    def update3DAll(self):
        return
if __name__ == "__main__":
    r = renderer3D()
    r.setup3DBest()