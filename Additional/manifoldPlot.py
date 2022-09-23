import re, seaborn as sns
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid", {'axes.grid' : False})


def plotManifold():
    fig = plt.figure(figsize=(6,6))
    x = []
    y = []
    z = []
    for i in range(len(matrices)):
        matrix = matrices[i]
        matrix.cpu()
        matrix = matrix.numpy()
        print(matrix)
        # Extract the submatrix
        SOsubmatrix = np.array([[matrix[0][0], matrix[0][1]]
                                [matrix[1][0], matrix[1][1]]])
        SOvector = np.reshape(SOsubmatrix, (4,1))
        x.append(float(SOvector[0]))
        y.append(float(SOvector[1]))
        z.append(float(SOvector[2]))
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
        
    #crop to a visualizable dimension
    SOvector = SOvector[0:2]
    
    #plotting
    ax = Axes3D(fig) # Method 1
    # ax = fig.add_subplot(111, projection='3d') # Method 2
    
    ax.scatter(x, y, z, c=x, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    

                    
                        
