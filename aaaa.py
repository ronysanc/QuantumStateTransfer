import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Setup inicial

g = 1.0        # <-- Energia de Hopping
gmin = 0.0    # <-- Energia de Hopping das bordas da cadeia
n = 30        # <-- Número de estados do sistema
t_total = 1500 # <-- Tempo total da simulação

# Define a função para evolução temporal dos estados
def time_evolution(H, psi0, t_list):
    n = H.shape[0]
    psi_t = np.zeros((len(t_list), n), dtype=complex)
    for i, t in enumerate(t_list):
        psi_t[i, :] = la.expm(-1j * H * t).dot(psi0)
    return psi_t

# Define a matriz Hamiltoniana
def MatrizTridiagonal(Principal, Secundaria):
  # Length da array diagonal
  N = len(Principal)
  # Matriz nula
  Matriz = np.zeros((N, N), dtype='complex')
  # Montar diagonais
  Matriz[0, 0] = Principal[0]
  for i in range(1, N):
    Matriz[i, i] = Principal[i]
    Matriz[i - 1, i] = Secundaria[i - 1]
    Matriz[i, i - 1] = np.conjugate(Secundaria[i - 1])
  return Matriz


def Main(gmin):
    DiagonalPrincipal = [0.0 for i in range(n)] # <-- Define os elementos da Diagonal Principal
    DiagonalSecundaria = [
    gmin if (i == 0 or i == n - 2) else g       # <-- Define os elementos da Diagonal Secundária
    for i in range(n - 1)]

    # Cria a matriz Hamiltoniana
    H = MatrizTridiagonal(DiagonalPrincipal, DiagonalSecundaria)

    # Definir o estado inicial
    psi0 = np.zeros(n)
    psi0[0] = 1

    # Definir o tempo e número de passos
    t_list = np.linspace(0, t_total, t_total*20)

    # Efetuar a evolução temporal dos estados
    psi_t = time_evolution(H, psi0, t_list)

    # Calcular a probabilidade de cada estado ao longo do tempo
    prob = np.abs(psi_t)**2

    # Armazenar a probabilidade x tempo de cada estado em um arquivo .dat
    #np.savetxt("prob_time.dat", np.hstack((t_list[:, np.newaxis], prob)))
    maxvalue = np.amax(prob[:,n-1])
    return maxvalue
with open("HoppingVariando.dat", "w") as Dados:
    for hopping in np.arange(gmin, 1.0, 0.05):
        valor = Main(hopping)
        Dados.write(f'{hopping:.2f} {valor}\n')

'''
# Plotar a probabilidade x tempo de cada estado
for i in range(n):
    if i == 0 or i==n-1: # <-- Condicional para plotar apenas os estados 1 e n
        plt.plot(t_list, prob[:, i], label="Estado {}".format(i+1))
plt.xlabel("Tempo")
plt.ylabel("Probabilidade")
plt.legend()
plt.show()
'''