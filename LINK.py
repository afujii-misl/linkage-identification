import numpy as np
import sys
import copy
from process import *
from sklearn.linear_model.tests.test_ridge import ind
sys.path.append("../deap_misl")
import random as rnd
import tools as misl_tools
import matplotlib.pyplot as plt
from matplotlib import cm
from deap import base, creator, tools
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt

def gen_epistasis(N, K):
    
    matrix = np.zeros((N**2, N, N))
    matrix2 = list()
     
    for h in range(N**2):
        raw = int(h/N)
        col = h%N
        
        locus = [rnd.choice(range(N)) for i in range(K+1)]
        locus2 =[rnd.choice(range(N)) for i in range(K+1)]
       
        if raw in locus and col == locus2[locus.index(raw)]: 
            pass
        
        else:
            locus[0] = raw
            locus2[0] = col
        
        epistasis = np.zeros((2, K+1))
        
        epistasis[0] = locus
        epistasis[1] = locus2
        
        matrix2.append(epistasis)

    return np.array(matrix2).astype(int)

def culc_fitness(genotype, rnt, epistasis, key):

    genotype =to_matrix(genotype) 
    genotype_np= np.array(genotype)
    
    N = len(genotype)
    fitness = np.zeros(N**2)
    
    for i in np.arange(N**2):
        fitness[i] = rnt[int(np.sum(genotype_np[epistasis[i, 0], epistasis[i, 1]] * key)), i]
    
    genotype = individual_flatten(genotype)
    
    return np.mean(fitness),

    
def individual_flatten(ind):
    
    ind_1=np.ravel(ind)

    [ind.pop(0) for i in range(len(ind))]
    [ind.append(i) for i in ind_1]
    
    return ind

def to_matrix(genotype):
    
    ind3=np.array(genotype)
    ind3=ind3.reshape(N,N)
    [genotype.pop(0) for i in range(len(genotype))]
    [genotype.append(list(i)) for i in ind3]
    return genotype

def perturb(ind,index):
    
    ind_p=copy.deepcopy(ind)
    ind_p[index] = 1-ind[index]
    return ind_p


def LINK(pop):
    
    error=0.001
    lincage_list=[[] for i in range(N*N)]
    
    for ind in tqdm(pop):
        ind2 = individual_flatten(ind)
        for i in range(N*N):
            p1=perturb(ind2,i)#perturb    
            df1 = toolbox.evaluate(p1)[0]-toolbox.evaluate(ind2)[0]
            
            for j in range(N*N):
                if i!=j:
                    p1=perturb(ind2,j)
                    df2=toolbox.evaluate(p1)[0]-toolbox.evaluate(ind2)[0]
                    p3=perturb(p1,i)       
                    df12=toolbox.evaluate(p3)[0]-toolbox.evaluate(ind2)[0]
                   
                    if abs(df12-(df1+df2))>error:
                        lincage_list[i].append(j)
                        lincage_list[j].append(i)
                        
                        
    link_unique = [list(set(i)) for i in lincage_list]        
    
    return  link_unique
     

def LIEM(pop):
    error=0.001
    e=np.ones((N*N,N*N))
    lincage_list=[[] for i in range(N*N)]
    
    #Calculate epistasis measure e[i][j]
    for i in range(N*N):
        for j in range(N*N):
            e[i][j]=0
            if i!=j:
                for ind in (pop):
                    ind2 = individual_flatten(ind)
                    p1 = perturb(ind2,i)
                    df1 = toolbox.evaluate(p1)[0]-toolbox.evaluate(ind2)[0]
                    p2 = perturb(ind2,j)
                    df2 = toolbox.evaluate(p2)[0]-toolbox.evaluate(ind2)[0]
                    p3 = perturb(p1,j)       
                    df12 = toolbox.evaluate(p3)[0]-toolbox.evaluate(ind2)[0]
                    
                    epistasis_measure = abs(df12-(df1+df2))
                    if epistasis_measure > e[i,j]:
                        e[i,j]=round(epistasis_measure,5)
    print np.array(e)
    
    #fix i and sorting e[i][j] with j
    for i in range(N*N):
        max=np.sort(e[i][:])[::-1]
        max_ind=np.argsort(e[i][:])[::-1]
        
    #select linkages
        for k in range(2):
            if max[k]>error:
                lincage_list[i].append(max_ind[k])
                
    return lincage_list

if __name__=='__main__':
    
    trial=100
    POP_SIZE = 10
    N_GEN =1000
    CXPB = 1.0
    N = 4
    K = 1
    key = np.power(2, np.arange(K, -1, -1))
    seed = 0
    rnd.seed(seed)
    np.random.seed(seed)
    rnt = np.random.randint(1, 101, (2**(K+1), N**2))
    epistasis = gen_epistasis(N, K)
  
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", rnd.randint, 0, 1)
    toolbox.register("attr_row", tools.initRepeat, list, toolbox.attr_bool, N)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_row, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", misl_tools.cxUniform1d_2)
    toolbox.register("mutate1d", tools.mutFlipBit, indpb=1.0/(N*N))
    toolbox.register("mutate", misl_tools.mutate2d, toolbox=toolbox)
    toolbox.register("select", tools.selTournament,tournsize=3)
    toolbox.register("evaluate",culc_fitness, rnt=rnt, epistasis=epistasis, key=key)
    
    pop = toolbox.population(n=POP_SIZE)
    
    print LINK(pop)
    print LIEM(pop)
    