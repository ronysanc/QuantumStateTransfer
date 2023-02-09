import numpy as np

# Gera a matriz Hamiltoniana
H = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

# Calcula os autovetores e autovalores
w, v = np.linalg.eigh(H)

# Define o tempo de evolução
t = np.linspace(0, 10, 1000)

# Calcula a evolução temporal de cada autoestado

for t in np.arange(0, 1000, 0.1):
  for i in range(3):
    psi = np.array([np.exp(-1j * w[i] * t) * v[:, i]])
  prob = np.abs(psi)**2
  print(prob)

'''
# Calcula a probabilidade de cada estado como função do tempo
prob = np.abs(psi)**2
print(prob)
# Grava o arquivo .dat com a probabilidade x tempo
with open("probabilidade_x_tempo.dat", "w") as f:
    for i in range(1000):
        f.write("{} {} {} {}\n".format(t[i], prob[0, i], prob[1, i], prob[2, i]))
'''