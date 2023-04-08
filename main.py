import numpy as np
import scipy.linalg as la
import sys

#import matplotlib.pyplot as plt

'''
import time
start_time = time.time()'''
# Setup inicial
'''
#np.random.seed(0)
g = 1.0       # <-- Energia de Hopping
gmin = 0.4    # <-- Energia de Hopping das bordas da cadeia
n = 50        # <-- Número de estados do sistema
t_total = 1000 # <-- Tempo total da simulação
alpha = 4.0   # <-- Alpha da distribuição correlacionada
'''
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

def Correlacao(N, alpha):
    phi = np.random.uniform(0, 2*np.pi, int(N/2)) # gerando números aleatórios phi_k
    omega = np.zeros(N) # inicializando o array omega com zeros

    for n in range(1, N+1):
        for k in range(1, int(N/2)+1):
            omega[n-1] += k**(-alpha/2)*np.cos(2*np.pi*n*k/N + phi[k-1])
            
    omega_mean = np.mean(omega) # calculando a média de omega
    omega_var = np.var(omega) # calculando a variância de omega
            
    omega = (omega - omega_mean)/np.sqrt(omega_var - omega_mean**2) # normalizando o array
    omega[0] = 0.0
    omega[N-1] = 0.0
    return omega

# Função Principal
def Main(gmin, dt):
    DiagonalPrincipal = Correlacao(N = n, alpha = alpha) # <-- Define os elementos da Diagonal Principal
    DiagonalSecundaria = np.array([
    gmin if (i == 0 or i == n - 2) else g       # <-- Define os elementos da Diagonal Secundária
    for i in range(n - 1)])

    # Cria a matriz Hamiltoniana
    H = MatrizTridiagonal(DiagonalPrincipal, DiagonalSecundaria)

    # Definir o estado inicial
    psi0 = np.zeros(n)
    psi0[0] = 1

    # Definir o tempo e número de passos
    t_list = np.linspace(0, t_total, dt)

    # Efetuar a evolução temporal dos estados
    psi_t = time_evolution(H, psi0, t_list)

    # Calcular a probabilidade de cada estado ao longo do tempo
    prob = np.abs(psi_t)**2

    # Armazenar a probabilidade x tempo de cada estado em um arquivo .dat
    #np.savetxt("prob_time.dat", np.hstack((t_list[:, np.newaxis], prob)))

    max_prob = np.max(prob[:, n-1]) # <-- Probabilidade máxima no último estado
    max_index = np.argmax(prob[:, n-1]) # <-- Índice do tempo em que a probabilidade é máxima
    max_time = t_list[max_index] # <-- Tempo em que a probabilidade é máxima
    
    '''
    # Plotar a probabilidade x tempo de cada estado
    for i in range(n):
        if i == 0 or i==n-1: # <-- Condicional para plotar apenas os estados 1 e n
            plt.plot(t_list, prob[:, i], label="Estado {}".format(i+1))
    plt.xlabel("Tempo")
    plt.ylabel("Probabilidade")
    plt.legend()
    plt.show()'''
    return max_prob, max_time

# Laço de repetição que escreve a probabilidade máxima do último nível variando o hopping mínimo

###############

n_amostra = int(sys.argv[1])
filename = '/home/rony/Documentos/Research/QuantumStateTransfer/Correlacao/Tempo-Alpha=4.0-HoppingVariandoCorrelacionado{:03d}.dat'.format(n_amostra)

with open(filename, "w") as Dados:
    gmin = 0.1    # <-- Energia de Hopping das bordas da cadeia
    g = 1.0       # <-- Energia de Hopping
    n = 50        # <-- Número de estados do sistema
    alpha = 4.0   # <-- Alpha da distribuição correlacionada
    for hopping in np.arange(gmin, 0.7, 0.01):
        if(hopping<0.4):
            t_total = 8000 # <-- Tempo total da simulação
            dt = t_total # <-- Espaçamento do linespace (dt = 1.0)
        else:
            t_total=500 # <-- Tempo total da simulação
            dt = t_total*2 # <-- Espaçamento do linespace (dt = 0.5)
        valor, tempo = Main(gmin=hopping, dt=dt)
        Dados.write(f'{hopping:.3f} {valor} {tempo}\n')

###############

#Main(gmin, t_total) # < -- Chama a função main 1x (para plotar o gráfico Prob x Tempo)
'''
end_time = time.time()
tempo_de_execucao = end_time - start_time
print('Tempo de execução: {:.3f} segundos'.format(tempo_de_execucao))'''