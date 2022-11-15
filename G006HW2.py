from re import S
import time
import sys
from math import sqrt
import os
import numpy as np

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
        
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return sqrt(res)

def SeqWeightedOutliers(P, W, k, z, alpha):
    
    length_P = len(P)
    W = np.array(W)
    
    # Make a list with all distances between points, avoiding 
    # calculating same distance twice (ij = ji):
    distances_list = [euclidean(P[i], P[j]) 
                      for i in range(length_P) 
                      for j in range(i+1, length_P)]

    # Build appropiate size triangular matrix and fill it
    # with distances:
    distances = np.zeros((length_P, length_P))
    distances[np.triu_indices(length_P, 1)] = distances_list
    # Make it a full matrix:
    distances = distances + distances.T

    # Calculate minimum distance between k+z+1 points
    # without counting same distances twice
    min_dist = min([distances[i][j] 
                    for i in range(k+z+1-1) 
                    for j in range(i+1, k+z+1)])

    r = min_dist/2
    print("Initial guess = ", r)
    counter = 1

    while 1:
        Z = list(range(length_P))
        S = []
        W_z = sum(W)
        r_inner = (1+2*alpha)*r
        r_outer = (3+4*alpha)*r

        while len(S) < k and W_z > 0:
            max = 0
            for indP, x in enumerate(P):
                # Sum the weights of all points inside the inner
                # radius of a given point in P:
                ball_weight = sum([W[indP] for indZ in Z 
                                   if distances[indP][indZ] <= r_inner])
                if ball_weight > max:
                    max = ball_weight
                    newcenter = x
                    new_c_ind = indP
            S.append(newcenter)
            # Substract from the total weight the sum of weights of the
            # points inside the outer radius (which are not considered outliers anymore):
            W_z = W_z - sum([W[indZ] for indZ in Z 
                             if distances[new_c_ind][indZ] <= r_outer])
            # Only keep as outliers those points out of the outer radius:
            Z = [indZ for indZ in Z 
                 if distances[new_c_ind][indZ] > r_outer]
        if W_z <= z:
            print("Final guess = ", r)
            print("Number of guesses = ", counter)
            return S
        else:
            r = 2*r
            counter += 1


def ComputeObjective(P,S,z):
    print(S)
    distances = [min([euclidean(point1,point2) 
                 for point2 in S]) 
                 for point1 in P]
    sorted_distances = sorted(distances)
    if z > 0:
        return sorted_distances[:-z][-1]
    else:
        return sorted_distances[-1]


def main():

    # INPUT READING
    assert len(sys.argv) == 4, \
           "Usage: python G006HW2.py <filename> <k> <z>"

    # Read data path
    data_path = sys.argv[1]
    assert os.path.isfile(data_path),"File or folder not found"

    # Read number of centers
    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    # Read number of outliers
    z = sys.argv[3]
    assert z.isdigit(), "z must be a number"
    z = int(z)

    inputPoints = readVectorsSeq(data_path)
    weights = [1] * len(inputPoints)

    print("Input size n = ", len(inputPoints))
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)

    t_before = time.process_time_ns()
    solution = SeqWeightedOutliers(inputPoints, weights, k, z, 0)
    t_after = time.process_time_ns()
    objective = ComputeObjective(inputPoints, solution, z)

    print("Objective function = ", objective)
    print("Time of SeqWeightOutliers = ", (t_after-t_before)/10**6)


if __name__ == "__main__":
    main()
