import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data_787.csv")

A = data["A"]
B = data["B"]
C = data["C"]
D = data["D"]

list = [A,B,C,D]
attributeNames = ["A","B","C","D"]

# Histograms
for i in list:
    print(i)
    #plt.hist(i)
    #plt.show()

data_matrix = data[attributeNames].to_numpy()

corr_matrix = np.corrcoef(data_matrix, rowvar=False)
print(corr_matrix)
#plt.matshow(corr_matrix)

N, M = data_matrix.shape
# Number of classes, C:
C = 4
# Correlation matrix visualized with scatter plot
fig1 = plt.figure(figsize=(18,15))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1*M + m2 + 1)
        plt.plot(np.array(data_matrix[:,m2]), np.array(data_matrix[:,m1]), '.')
        if m1==M-1:
            plt.xlabel(attributeNames[m2])
        else:
            plt.xticks([])
        if m2==0:
            plt.ylabel(attributeNames[m1])
        else:
            plt.yticks([])
        #ylim(0,X.max()*1.1)
        #xlim(0,X.max()*1.1)
#plt.legend(classNames, ncol = 6, loc = "lower center")
# Export
#plt.savefig('Correlation.png')
#plt.show()

# Conditioning on A
cond_A = data_matrix[A>1,:]
print(cond_A)
print(cond_A.shape)
"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(cond_A[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(cond_A[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(cond_A[:,i]))
"""
data_intervene_A2 = pd.read_csv("data_811.csv")
data_intervene_A2_matrix = data_intervene_A2[attributeNames].to_numpy()
"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(data_intervene_A2_matrix[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(data_intervene_A2_matrix[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(data_intervene_A2_matrix[:,i]))
"""
# Conditioning on D
cond_D = data_intervene_A2_matrix[data_intervene_A2_matrix[:,3]>2,:]
print(cond_D)
print(cond_D.shape)
"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(cond_D[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(cond_D[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(cond_D[:,i]))
"""

#Intervene B=2
data_intervene_B2 = pd.read_csv("data_827.csv")
data_intervene_B2_matrix = data_intervene_B2[attributeNames].to_numpy()
"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(data_intervene_B2_matrix[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(data_intervene_B2_matrix[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(data_intervene_B2_matrix[:,i]))
"""
# Conditioning on A
cond_A_B2 = data_intervene_B2_matrix[data_intervene_B2_matrix[:,0]>3,:]
print(cond_A_B2)
print(cond_A_B2.shape)

"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(cond_A_B2[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(cond_A_B2[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(cond_A_B2[:,i]))
"""

# Intervene C = -2
data_intervene_C2 = pd.read_csv("data_845.csv")
data_intervene_C2_matrix = data_intervene_C2[attributeNames].to_numpy()
"""
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(data_intervene_C2_matrix[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_matrix[:,i]))
    print(i,np.mean(data_intervene_C2_matrix[:,i]))
    print(i,np.var(data_matrix[:,i]))
    print(i,np.var(data_intervene_C2_matrix[:,i]))
"""

# Conditioning on B
cond_B_C2 = data_intervene_C2_matrix[data_intervene_C2_matrix[:,1]>1,:]
print(cond_B_C2)
print(cond_B_C2.shape)
"""
# Sammenligner (intervene C=-2) og (intervene C=-2 og B>1)
for i in range(4):
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(data_intervene_C2_matrix[:,i])
    plt.subplot(1,2,2)
    plt.hist(cond_B_C2[:,i])
    plt.show()
    #plt.hist(i)
    #plt.show()
    print(i,np.mean(data_intervene_C2_matrix[:,i]))
    print(i,np.mean(cond_B_C2[:,i]))
    print(i,np.var(data_intervene_C2_matrix[:,i]))
    print(i,np.var(cond_B_C2[:,i]))
"""
