'''
Created on Nov 14, 2013

@author: alecmacrae
'''
# project 3: MD

# input filename should not contain "/" or "." characters

# imports libraries and establishes the arguments that should be used to run the script
import sys
import os
import numpy
import time
import scipy
import scipy.spatial

argvMap = {"--kB":"40000.0",
           "--kN":"400.0",
           "--nbCutoff":"0.50",
           "--m":"12.0",
           "--dt":"0.001",
           "--n":"1000"}
for i in xrange(len(sys.argv)):
    if i%2 == 1:
        if sys.argv[i] == "--iF":
            argvMap["--iF"] = sys.argv[i+1]
            outputFileName = os.path.split(sys.argv[i+1])[-1]
            outputFileName = os.path.splitext(outputFileName)[-2]
            argvMap["--out"] = outputFileName
        else:
            argvMap[sys.argv[i]] = sys.argv[i+1]

# FUNCTIONS-----------------------------------------------------------

def getDistMatrix(r):
    """creates a distance matrix for real time lookup of distances"""
    tempMatrix = []
    for key in r:
        tempMatrix.append(r[key])
    distMatrix = scipy.spatial.distance.pdist(tempMatrix,"euclidean")
    distMatrixSquareForm = scipy.spatial.distance.squareform(distMatrix)
    return distMatrixSquareForm

def calculateBondLengths(connections,distMatrix):
    """calculates the initial bond lengths between bonded atoms"""
    d0Map = {}
    for key in connections:
        tempMap = {}
        for value in connections[key]:
            length = distMatrix[key-1,value-1] 
            tempMap[value] = length
        d0Map[key] = tempMap
    return d0Map

def calculateNonBondLengths(r,distMatrix):
    """calculates the initial non-bond lengths between close atoms"""
    r0Map = {}
    nbCutoff = float(argvMap["--nbCutoff"])
    for key1 in r:
        tempMap = {}
        for key2 in r:
            if key1 != key2:
                length = distMatrix[key1-1,key2-1] 
                if length < nbCutoff:
                    tempMap[key2] = length
        r0Map[key1] = tempMap
    return r0Map

def makeStartingCalculations():
    """calculates the initial velocities, positions, and bond and non-bond lengths from the input file"""
    v = {}
    r = {}
    connections = {}
    input_file = open(argvMap["--iF"])
    lineNumber = 1
    for line in input_file:
        if lineNumber == 1:
            argvMap["name"] = line.strip().split()[1]
            argvMap["T"] = line.strip().split()[-1]
        else:
            line = line.strip().split()
            line = map(float,line)
            v[line[0]] = [line[4],line[5],line[6]]
            r[line[0]] = [line[1],line[2],line[3]]
            if len(line) > 7:
                connections[line[0]] = line[7:]
        lineNumber+=1
    distMatrix = getDistMatrix(r)
    b0s = calculateBondLengths(connections,distMatrix)
    r0s = calculateNonBondLengths(r,distMatrix)
    startingCalcs = [b0s,r0s,v,r]
    input_file.close()
    return startingCalcs

def calculateForce(distMatrix,r0s,key1,key2,k):
    """calculates the force from one atom on another atom"""
    length1 = distMatrix[key1-1,key2-1] 
    length2 = numpy.array(r0s[key1][key2])
    force = k*(length1-length2)
    return force

def updateF(F,key1,key2,force):
    """updates the data structure F based on if F already has a particular key or not"""
    if F.has_key(key1):
        F[key1][key2] = force
    else:
        F[key1] = {}
        F[key1][key2] = force
    if F.has_key(key2):
        F[key2][key1] = force
    else:
        F[key2] = {}
        F[key2][key1] = force
    return F

def findForces(b0s,r0s,distMatrix):
    """finds force of an atom acting on another atom: mapping to return ID1->ID2->force"""
    kB = float(argvMap["--kB"])
    kN = float(argvMap["--kN"])
    F = {}
    for key1 in r0s:
        for key2 in r0s[key1]:
            if key1 < key2:
                if b0s.has_key(key1) and b0s[key1].has_key(key2): 
                    force = calculateForce(distMatrix,b0s,key1,key2,kB)
                    F = updateF(F,key1,key2,force)
                elif r0s.has_key(key1) and r0s[key1].has_key(key2):
                    force = calculateForce(distMatrix,r0s,key1,key2,kN)
                    F = updateF(F,key1,key2,force)
    return F

def breakForcesIntoComponents(F,r,distMatrix):
    """breaks up force into x,y,z components: mapping is ID1->ID2->[Fx,Fy,Fz]"""
    for key1 in F:
        for key2 in F[key1]:
            if key1 < key2:
                length = distMatrix[key1-1,key2-1] 
                force = F[key1][key2]
                Fx = force*(r[key2][0]-r[key1][0])/length
                Fy = force*(r[key2][1]-r[key1][1])/length
                Fz = force*(r[key2][2]-r[key1][2])/length
                F[key1][key2] = [Fx,Fy,Fz]
                F[key2][key1] = [-Fx,-Fy,-Fz]
    return F

def sumForceComponents(Fxyz):
    """sums forces in each component together for each atom: mapping is ID1->[Fx,Fy,Fz]"""
    FxyzSum = {}
    for key1 in Fxyz:
        x = 0
        y = 0
        z = 0
        for key2 in Fxyz[key1]:
            x += Fxyz[key1][key2][0]
            y += Fxyz[key1][key2][1]
            z += Fxyz[key1][key2][2]
        FxyzSum[key1] = [x,y,z]
    return FxyzSum

def updateForces(r,b0s,r0s):
    """updates the current forces in each direction for each atom"""
    distMatrix = getDistMatrix(r)
    F = findForces(b0s,r0s,distMatrix)
    Fxyz = breakForcesIntoComponents(F,r,distMatrix)
    FxyzSum = sumForceComponents(Fxyz)
    return FxyzSum

def updateAcceleration(m,F):
    """updates the acceleration of the atoms based on the forces"""
    a = {}
    for key in F:
        a[key] = [x/m for x in F[key]]
    return a

def updateVelocities(v,a,dt):
    """ updates the velovity of the atoms based on v+1/2at"""
    vNew = {}
    for key in v:
        at = [0.5*x*dt for x in a[key]]
        vNew[key] = [x+y for x,y in zip(v[key],at)]
    return vNew

def updatePositions(r,v,dt):
    """updates the positions of the atoms based on r+v*dt"""
    rNew = {}
    for key in r:
        vt = [x*dt for x in v[key]]
        rNew[key] = [x+y for x,y in zip(r[key],vt)]
    return rNew

def calculatePotentialEnergy(distMatrix,r0s,key1,key2,k):
    """calculates the potential energy between 2 atoms"""
    length1 = distMatrix[key1-1,key2-1] 
    length2 = numpy.array(r0s[key1][key2])
    PE = 0.5*k*((length1-length2)**2)
    return PE

def updatePotentialEnergy(r,b0s,r0s):
    """calculates the bonded and non-bonded potential energy over all of the atoms"""
    distMatrix = getDistMatrix(r)
    kB = float(argvMap["--kB"])
    kN = float(argvMap["--kN"])
    PEbonds = 0
    PEnonbonds = 0
    for key1 in r0s:
        for key2 in r0s[key1]:
            if key1 < key2:
                if b0s.has_key(key1) and b0s[key1].has_key(key2):
                    PEbonds += calculatePotentialEnergy(distMatrix,b0s,key1,key2,kB)
                else:
                    PEnonbonds += calculatePotentialEnergy(distMatrix,r0s,key1,key2,kN)
    return [PEbonds,PEnonbonds]

def calculateKineteicEnergy(m,v):
    """calculates the kinetic energy at a given time frame using 1/2mv^2"""
    KE = 0
    for key in v:
        for value in v[key]:
            KE += 0.5*m*(value**2)
    return KE

def writeERG(i,PE,KE,Etotal):
    """writes to a .erg file to be output"""
    if i == 10:
        erg.write("# step\tE_k\tE_b\tE_nB\tE_tot\n")
    erg.write("%i\t%.1f\t%.1f\t%.1f\t%.1f\n" % (i,KE,PE[0],PE[1],Etotal))

def writeRVC(i,r,v,b0s,Etotal):
    """writes to a .rvc file to be output"""
    if i == 0:
        rvc.write("# %s\tkB=%s\tkN=%s\tnbCutoff=%s\tdt=%s\tmass=%s\t%s\n" % 
                  (argvMap["name"],argvMap["--kB"],argvMap["--kN"],argvMap["--nbCutoff"],
                   argvMap["--dt"],argvMap["--m"],argvMap["T"]))
    else:
        rvc.write("#At time step %i,energy = %.3fKJ\n" % (i,Etotal))
    for key in r:
        rvc.write("%i\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % 
                  (key,r[key][0],r[key][1],r[key][2],v[key][0],v[key][1],v[key][2]))
        if b0s.has_key(key):
            for key2 in b0s[key]:
                rvc.write("\t%i" % (key2))
        rvc.write("\n")

def writeOutput(i,PE,KE,r,v,b0s):
    """writes 2 outputs. A .erg and a .rvc file"""
    Etotal = PE[0]+PE[1]+KE
    writeERG(i,PE,KE,Etotal)
    writeRVC(i,r,v,b0s,Etotal)

def iterateTimesteps(startingCalcs):
    """loops through a number of timesteps and recalculates position and velocity 
    for each timestep. Writes an output every 10 timesteps"""
    EtotalInitial = 0
    m = float(argvMap["--m"])
    dt = float(argvMap["--dt"])
    b0s = startingCalcs[0]
    r0s = startingCalcs[1]
    v = startingCalcs[2]
    r = startingCalcs[3]
    F = updateForces(r,b0s,r0s)
    a = updateAcceleration(m,F)
    writeRVC(0,r,v,b0s,0)
    for i in xrange(int(argvMap["--n"])):
        v = updateVelocities(v,a,dt)
        r = updatePositions(r,v,dt)
        PE = updatePotentialEnergy(r,b0s,r0s)
        F = updateForces(r,b0s,r0s)
        a = updateAcceleration(m,F)
        v = updateVelocities(v,a,dt)
        KE = calculateKineteicEnergy(m,v)        
        if (i+1)%10 == 0:
            writeOutput(i+1,PE,KE,r,v,b0s)
        
        # checks for overflow possibility    
        Etotal = KE+PE[0]+PE[1]
        if i == 0:
            EtotalInitial = KE+PE[0]+PE[1]
        elif Etotal > 10*EtotalInitial:
            print "Warning, Total Energy has grown above 10 times the initial total energy"
        elif Etotal < EtotalInitial/10:
            print "Warning, Total Energy has gone below 10 times the initial total energy"        

def run():
    """runs the program by initializing parameters and then iterating over the timesteps"""
    startingCalcs = makeStartingCalculations()
    iterateTimesteps(startingCalcs)

#--OUTPUT-------------------------------------------------

output_file1 = argvMap["--out"] + "_out.erg"
erg = open(output_file1, 'w')
output_file2 = argvMap["--out"] + "_out.rvc"
rvc = open(output_file2, "w")            
run()
erg.close()
rvc.close()
