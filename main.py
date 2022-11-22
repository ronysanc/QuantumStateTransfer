import cmath
import numpy as np
from numpy import linalg as LA


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


if __name__ == '__main__':
  NumerodeElementos = 4
  SitioInicial = 0
  TempoSimulado = 300.0
  dt = 0.1
  #Configurações de estado inicial
  EstadoInicial = [
    1.0 if (i == SitioInicial) else 0.0 for i in range(NumerodeElementos)
  ]
  DiagonalPrincipal = [0.0 for i in range(NumerodeElementos)]
  DiagonalSecundaria = [
    0.001 if (i == 0 or i == NumerodeElementos - 2) else 1.0
    for i in range(NumerodeElementos - 1)
  ]
  #Montar a matriz hamiltoniana
  MatrizHamiltoniana = MatrizTridiagonal(DiagonalPrincipal, DiagonalSecundaria)
  #Diagonalizar a matriz
  AutoValor, AutoVetor = LA.eig(np.array(MatrizHamiltoniana))
  #Computar a evolução temporal
  with open("EvolucaoTemporal.dat", "w") as Dados:
    #Calcular autodecomposição do estado inicial
    AutoComponente = []
    for i in range(NumerodeElementos):
      Componente = 0.0 + 0.0j
      for l in range(NumerodeElementos):
        Componente += EstadoInicial[l] * np.conjugate(AutoVetor[i][l])
      AutoComponente.append(Componente)
    #Calcular evolução temporal na base dos autoestados
    for iTempo in range(0, int(TempoSimulado / dt)):
      Tempo = iTempo * dt
      Dados.write(f"{Tempo:.1f} ")
      for i in range(NumerodeElementos):
        Componente = 0.0 + 0.0j
        for l in range(NumerodeElementos):
          Componente += cmath.exp(
            -1.0j * AutoValor[l] * Tempo) * AutoComponente[l] * AutoVetor[l][i]
        Probabilidade = Componente * np.conjugate(Componente)
        Dados.write(f"{Probabilidade.real} ")
      Dados.write("\n")
