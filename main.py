import cmath
import numpy as np
from numpy import linalg as LA
'''
B = np.matrix('[1, 0, 0; 0, 1, 0; 0, 0, 2]')
#print(B.getI()*B)
#print(np.dot(B.getI(),B))

class Hamiltonian:
  def __init__(self, matrix):
    self.matrix = matrix
    self.transpose = matrix.T

#print(B[1,1])

#Hamiltoniana = Hamiltonian( matrix = B )
'''

#SETUP:
N=7 #Quantidade de niveis
tMax=500 #Tempo de simulacao 
dt=0.1 #Time step
g=1.
gMin=0.001

#Alocar matrix Hamiltoniana NxN
Hamiltoniana=np.zeros((N,N)) # np.zeros((N,N), dtype = complex) for complex matrix
#Gerar matrix Hamiltoniana
for i in range(N):
    for j in range(N):
        if i==j: #local energy
            Hamiltoniana[i,j]=0.
        elif i==j+1 or i==j-1: #hopping energy
            if (j==0 and i==1) or (j==1 and i==0):
                Hamiltoniana[i,j]=gMin
            elif (j==N-1 and i==N-2) or (j==N-2 and i==N-1):
                Hamiltoniana[i,j]=gMin
            else:
                Hamiltoniana[i,j]=1.
        else: #others
            Hamiltoniana[i,j]=0.

print(Hamiltoniana)

EigenValues, EigenVectors = LA.eig(np.array(Hamiltoniana))

print(f'EigenValues:\n {EigenValues}\n')
print(f'EigenVectors:\n {EigenVectors}\n')


print(EigenValues[0])
print(EigenVectors[0])
#Funcao calcular probabilidade do sitio
def probabilidade(i, Psi):
    densidadeProbabilidade = 0j
    for j in range(N):
        densidadeProbabilidade += Psi[j]*EigenVectors[j,i]
    return (densidadeProbabilidade.real)**2+(densidadeProbabilidade.imag)**2

#Estado inicial da rede
EstadoInicial=np.zeros(N, dtype='complex')
EstadoInicial[0]=1.0+0j

#computar o estado inicial com base nos autoestados
EstadoAutodecomposicao=np.zeros(N, dtype='complex')
for i in range(N):
    for j in range(N):
        EstadoAutodecomposicao[i]+=np.conjugate(EigenVectors[i,j])*EstadoInicial[j]

file01 = open('dados.out','w')
for t in np.arange(0, tMax, step=dt):
    for i in range(N):
        EstadoAutodecomposicao[i] *= cmath.exp(-1j*EigenValues[i]*dt)
    file01.write(f'{t} ')

    for i in range(N):
        prob = probabilidade(i,EstadoAutodecomposicao)
        file01.write(f'{prob} ')
    file01.write('\n')

file01.close()


'''
for i in np.arange(0, 4.5, 0.5):
    if i != 4.0:
        print(i, end=', ')
    else:
        print(i)
#print(f'\n{np.exp(A)}')

# teste
D=np.zeros((N,N))
P=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i == j:
            D[i,j]=EigenValues[i]
            P[i,j]=EigenVectors[i,j]
        else:
            D[i,j]=0.
            P[i,j]=EigenVectors[i,j]

print(D)
print('-----------------')
J=np.matrix('1, 2, 3; 4, 5, 6; 7, 8, 10')
print(J)
print('-----------------')
print(np.linalg.inv(P)*D*P)

estadoAutodecomposicao=np.zeros(N, dtype='complex')
estadoAutodecomposicao[0]=1.



for t in np.arange(0, 1, step=0.1):
    for i in range(N):
        estadoAutodecomposicao[i] *= cmath.exp(-1j*EigenValues[i]*0.1)
        print(estadoAutodecomposicao[i])
'''
