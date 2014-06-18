'''
Created on Nov 20, 2013

@author: alecmacrae
'''
import sys
import scipy
import scipy.spatial

input_file1 = sys.argv[1]
input_file2 = sys.argv[2]
input_file3 = sys.argv[3]
output_file1 = sys.argv[4]
output_file2 = sys.argv[5]
f1 = open(input_file1)
f2 = open(input_file2)
f3 = open(input_file3)

def getPoints(f):
    iteration = -10
    atom17 = {}
    atom96 = {}
    atom298 = {}
    atom385 = {}
    for line in f:
        line = line.strip().split()
        if line[0] == "1":
            iteration += 10
        elif line[0] == "17":
            atom17[iteration] = [line[1],line[2],line[3]]
        elif line[0] == "96":
            atom96[iteration] = [line[1],line[2],line[3]]
        elif line[0] == "298":
            atom298[iteration] = [line[1],line[2],line[3]]
        elif line[0] == "385":
            atom385[iteration] = [line[1],line[2],line[3]]
    return [atom17,atom96,atom298,atom385]
        
def getDistMatrix(a,b):
    tempMatrix = []
    steps = range(0,1001,10)
    for i in steps:
        tempMatrix.append(a[i])
        tempMatrix.append(b[i])
    distMatrix = scipy.spatial.distance.pdist(tempMatrix,"euclidean")
    distMatrixSquareForm = scipy.spatial.distance.squareform(distMatrix)
    return distMatrixSquareForm
        
def calculateDistances(atoms1,atoms2,atoms3):
    distMatrix1 = getDistMatrix(atoms1[0],atoms1[1])
    distMatrix2 = getDistMatrix(atoms1[2],atoms1[3])
    
    distMatrix3 = getDistMatrix(atoms2[0],atoms2[1])
    distMatrix4 = getDistMatrix(atoms2[2],atoms2[3])
    
    distMatrix5 = getDistMatrix(atoms3[0],atoms3[1])
    distMatrix6 = getDistMatrix(atoms3[2],atoms3[3])
    
    f1 = open(output_file1, 'w')
    f2 = open(output_file2, 'w')
    steps = range(0,101)
    for i in steps:
        dist1 = distMatrix1[i][i+101]
        dist2 = distMatrix2[i][i+101]
        dist3 = distMatrix3[i][i+101]
        dist4 = distMatrix4[i][i+101]
        dist5 = distMatrix5[i][i+101]
        dist6 = distMatrix6[i][i+101]
        f1.write("%i\t%.4f\t%.4f\t%.4f\n" % (i*10,dist1,dist3,dist5))
        f2.write("%i\t%.4f\t%.4f\t%.4f\n" % (i*10,dist2,dist4,dist6))
    f1.close()
    f2.close()
        
atoms1 = getPoints(f1)
f1.close()
atoms2 = getPoints(f2)
f2.close()
atoms3 = getPoints(f3)
f3.close()
distances = calculateDistances(atoms1,atoms2,atoms3)
        