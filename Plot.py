import matplotlib.pyplot as plt
import numpy as np

#with open("HoppingVariando.dat", "r") as Arq:
filename = 'HoppingVariando.dat'

# Lê as colunas do arquivo .dat e armazena nas variáveis x e y
x, y = np.loadtxt(filename, unpack=True)

# Mostra os arrays x e y
print('x:', x)
print('y:', y)

plt.plot(x, y, '--o',color='black' , label='Probabelidade máxima')
plt.title('N=30, T=1500', loc='center')
plt.xlabel("Hopping mínimo", fontsize=14)
plt.ylabel("Probabilidade", fontsize=14)
plt.legend(loc=4)
plt.show()