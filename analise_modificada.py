# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt



def estampa_Resistor (M, no_a, no_b, valor):
    M[no_a,no_a] += 1/valor
    M[no_a,no_b] -= 1/valor
    M[no_b,no_a] -= 1/valor
    M[no_b,no_b] += 1/valor
    
    return M




def estampa_Diodo(M, no_a, no_b, corrente, tensao):

    G = ((corrente * (np.exp(1 / tensao)))/tensao)
    M[no_a, no_a] += G
    M[no_a, no_b] -= G
    M[no_b, no_a] -= G
    M[no_b, no_b] += G

    return M




def estampa_Capacitor (M, no_a, no_b, valor, frequencia):
    M[no_a,no_a] += 1j*frequencia*valor
    M[no_a,no_b] -= 1j*frequencia*valor
    M[no_b,no_a] -= 1j*frequencia*valor
    M[no_b,no_b] += 1j*frequencia*valor
    
    return M




def estampa_Indutor (M, tensao, no_a, no_b, valor, frequencia):
    
    dimensao = M.shape[0]

    linha = np.zeros(dimensao)
    M = np.append(M, [linha], axis=0)
    M[dimensao,no_a] -= 1
    M[dimensao,no_b] += 1
    
    coluna = np.zeros((dimensao+1,1))
    M = np.hstack((M, coluna))
    M[no_a,dimensao] += 1
    M[no_b,dimensao] -= 1
    M[dimensao,dimensao] += 1j*frequencia*valor

    tensao = np.append(tensao, 0)

    return M,tensao




def estampa_Transformador (M, tensao, no_a, no_b, no_c, no_d, L1, L2, MI, frequencia):
    dimensao = M.shape[0]
    linha = np.zeros(dimensao)
    M = np.append(M, [linha, linha], axis=0)
    M[dimensao, no_a] -= 1
    M[dimensao, no_b] += 1
    M[dimensao+1, no_c] -= 1
    M[dimensao+1, no_d] += 1

    coluna = np.zeros((dimensao+2,1))
    M = np.hstack((M,coluna, coluna))
    M[no_a,dimensao] += 1
    M[no_b,dimensao] -= 1
    M[no_c,dimensao+1] += 1
    M[no_d,dimensao+1] -= 1
    M[dimensao,dimensao] += 1j*frequencia*L1
    M[dimensao+1,dimensao] += 1j*frequencia*MI
    M[dimensao,dimensao+1] += 1j*frequencia*MI
    M[dimensao+1,dimensao+1] += 1j*frequencia*L2

    tensao = np.append(tensao, [0,0])
    return M, tensao



    
def estampa_FonteI_DC (M, no_a, no_b, valor):
    M[no_a] -= valor
    M[no_b] += valor
    
    return M




def estampa_FonteI_AC (M, no_a, no_b, amplitude, fase):
    M[no_a] -= amplitude*np.exp(1j*fase)
    M[no_b] += amplitude*np.exp(1j*fase)
    return M




def estampa_FonteV_DC (M, tensao,no_a, no_b, valor):  
    dimensao = M.shape[0]
    
    linha = np.zeros(dimensao)
    M = np.append(M, [linha], axis=0)
    M[dimensao,no_a] -= 1
    M[dimensao,no_b] += 1
    
    coluna = np.zeros((dimensao+1,1))
    M = np.hstack((M, coluna))
    M[no_a,dimensao] += 1
    M[no_b,dimensao] -= 1
    
    tensao = np.append(tensao, -valor)

    return M, tensao





def estampa_FonteV_AC (M, tensao, no_a, no_b, amplitude, fase):
    dimensao = M.shape[0]
    
    linha = np.zeros(dimensao)
    M = np.append(M, [linha], axis=0)
    M[dimensao,no_a] -= 1
    M[dimensao,no_b] += 1
    
    coluna = np.zeros((dimensao+1,1))
    M = np.hstack((M, coluna))#, axis=1)
    M[no_a,dimensao] += 1
    M[no_b,dimensao] -= 1
    
    tensao = np.append(tensao, -amplitude*np.exp(1j*fase))

    return M,tensao




    
def estampa_FonteE (M,tensao,no_a, no_b, no_c, no_d,valor):   
    dimensao = M.shape[0]
    
    linha = np.zeros(dimensao)
    M = np.append(M, [linha], axis=0)
    M[dimensao,no_a] -= 1
    M[dimensao,no_b] += 1
    M[dimensao,no_c] += valor
    M[dimensao,no_d] -= valor
    
    
    coluna = np.zeros((dimensao+1,1))
    M = np.hstack((M, coluna))
    M[no_a,dimensao] += 1
    M[no_b,dimensao] -= 1
    
    tensao = np.append(tensao, 0)

    return M, tensao



    
def estampa_FonteF (M,tensao,no_a,no_b,no_c,no_d,valor):
    dimensao = M.shape[0]
    
    linha = np.zeros(dimensao)
    M = np.append(M, [linha], axis=0)
    M[dimensao,no_c] -= valor
    M[dimensao,no_d] += valor
    
    newCol = np.zeros((dimensao+1,1))
    M = np.hstack((M, newCol))
    M[no_a,dimensao] += valor
    M[no_b,dimensao] -= valor
    M[no_c,dimensao] += 1
    M[no_d,dimensao] -= 1
    
    tensao = np.append(tensao, 0)
    
    return M, tensao




def estampa_FonteG (M,no_a, no_b, no_c, no_d, valor):
    M[no_a, no_c] += valor
    M[no_a, no_d] -= valor
    M[no_b, no_c] -= valor
    M[no_b, no_d] += valor
    
    return M




def estampa_FonteH (M,tensao,no_a,no_b,no_c,no_d,valor):   
    dimensao = M.shape[0]
    linha = np.zeros(dimensao)
    M = np.append(M, [linha, linha], axis=0)
    M[dimensao, no_c] -= 1
    M[dimensao, no_d] += 1
    M[dimensao+1,no_a] -= 1
    M[dimensao+1, no_b] += 1
    
    coluna = np.zeros((dimensao+2,1))
    M = np.hstack((M, coluna, coluna))
    M[no_c,dimensao] += 1
    M[no_d,dimensao] -= 1
    M[no_a,dimensao+1] += 1
    M[no_b,dimensao+1] -= 1
    M[dimensao+1, dimensao] += valor
    
    tensao = np.append(tensao, [0,0])

    return M, tensao





def estampa_componenteDC(M, tensao, no_a, no_b, no_c, no_d, no_e, no_f, K1, K2, K3, K4, K5, K6, K7, K8, I1, I2):    

    dimensao = M.shape[0]
    while dimensao <= max(no_a, no_b, no_c, no_d, no_e, no_f):
        M = np.pad(M, ((0, 1), (0, 1)), 'constant')
        tensao = np.append(tensao, 0)
        dimensao += 1
    
    M[no_a, no_a] += K1
    M[no_a, no_b] -= K1
    M[no_a, no_c] += K2
    M[no_a, no_d] -= K2
    M[no_a, no_e] += K3
    M[no_a, no_f] -= K3
    
    M[no_b, no_a] -= K1
    M[no_b, no_b] += K1
    M[no_b, no_c] -= K2
    M[no_b, no_d] += K2
    M[no_b, no_e] -= K3
    M[no_b, no_f] += K3
    
    M[no_c, no_c] += K5
    M[no_c, no_d] -= K5
    M[no_c, no_e] += K6
    M[no_c, no_f] -= K6
    
    M[no_d, no_c] -= K5
    M[no_d, no_d] += K5
    M[no_d, no_e] -= K6
    M[no_d, no_f] += K6
    
    M[no_e, no_e] += K8
    M[no_e, no_f] -= K8
    
    M[no_f, no_e] -= K8
    M[no_f, no_f] += K8
    
    tensao[no_a] += I1
    tensao[no_b] -= I1
    tensao[no_e] += I2
    tensao[no_f] -= I2
    
    return M, tensao





def estampa_componenteAC(M, tensao, no_a, no_b, no_c, no_d, no_e, K1, K2, K3, K4, K5, K6, frequencia):
    dimensao = M.shape[0]
    while dimensao <= max(no_a, no_b, no_c, no_d, no_e):
        M = np.pad(M, ((0, 1), (0, 1)), 'constant')
        tensao = np.append(tensao, 0)
        dimensao += 1

    # Primeira equação: Iab = K1 * Vab + jωK2 * Vcb + K3 * Ide
    M[no_a, no_a] += K1
    M[no_a, no_b] -= K1
    M[no_a, no_c] -= 1j*frequencia*K2
    M[no_a, no_b] += 1j*frequencia*K2
    M[no_a, no_d] -= K3
    M[no_a, no_e] += K3
    
    M[no_b, no_a] -= K1
    M[no_b, no_b] += K1
    M[no_b, no_c] += 1j*frequencia*K2
    M[no_b, no_b] -= 1j*frequencia*K2
    M[no_b, no_d] += K3
    M[no_b, no_e] -= K3

    # Segunda equação: Icb = jω * Vab + K4 * Vcb
    M[no_c, no_a] -= 1j*frequencia
    M[no_c, no_b] += 1j*frequencia
    M[no_c, no_c] += K4
    M[no_c, no_b] -= K4
    
    M[no_b, no_a] += 1j*frequencia
    M[no_b, no_b] -= 1j*frequencia
    M[no_b, no_c] -= K4
    M[no_b, no_b] += K4

    # Terceira equação: Vde = K5 * Vab + K6 * Vcb
    # Adicionar novas linhas e colunas
    linha = np.zeros(dimensao)
    M = np.append(M, [linha, linha], axis=0)
    M[dimensao, no_a] -= K5
    M[dimensao, no_b] += K5
    M[dimensao+1, no_c] -= K6
    M[dimensao+1, no_b] += K6

    coluna = np.zeros((dimensao+2,1))
    M = np.hstack((M, coluna, coluna))
    M[no_a,dimensao] += K5
    M[no_b,dimensao] -= K5
    M[no_c,dimensao+1] += K6
    M[no_b,dimensao+1] -= K6

    tensao = np.append(tensao, [0, 0])
    
    return M, tensao




    
def lerArquivo(arquivo_netlist):
    no_maior=0
    ArqNetlist = []
    componentes = []
    
    with open(arquivo_netlist, "r") as arquivo:
        for linha in arquivo:
            linha = linha.strip()  # Remover espaços em branco no início e no final da linha
            if linha and not linha.startswith('*'):  # Ignorar linhas em branco e comentários

                
                termo = linha.split(" ")
                tupla=tuple(termo)
                ArqNetlist.append(tupla)    #Lista com todos os termos e valores
                tipo_elemento = linha[0]    #Tipo do componente

            
                #Definicao dos maiores nos
            
                if tipo_elemento[0] == "R":

                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    valor = float(termo[3])

                    if max(no_a, no_b) > no_maior:
                    
                        no_maior = max(no_a, no_b)
                    componentes.append(['R',no_a,no_b,valor])




                elif tipo_elemento[0] == "D":

                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    corrente = float(termo[3])
                    tensao = float(termo[4])
   
                    
                    if max(no_a,no_b) > no_maior:
                    
                        no_maior = max(no_a,no_b)
                    componentes.append(['D',no_a,no_b,corrente,tensao])



                elif tipo_elemento == 'C':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    valor = float(termo[3])
                
                    if max(no_a,no_b) > no_maior:

                        no_maior = max(no_a,no_b)
                    componentes.append(['C',no_a,no_b,valor])



                elif tipo_elemento == 'L':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    valor = float(termo[3])
                
                    if max(no_a,no_b) > no_maior:

                        no_maior = max(no_a,no_b)
                    componentes.append(['L',no_a,no_b,valor])





                elif tipo_elemento == 'K':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    L1 = float(termo[5])
                    L2 = float(termo[6])
                    M = float(termo[7])
                
                    if max(no_a,no_b,no_c,no_d) > no_maior:
                    
                        no_maior = max(no_a,no_b,no_c,no_d)
                    componentes.append(['K',no_a,no_b,no_c,no_d,L1,L2,M])







                elif tipo_elemento == 'I': 
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    tipo_simulacao = termo[3]
                
                    if tipo_simulacao == 'DC':
                        valor = float(termo[4])
                        componentes.append(['I',no_a,no_b,tipo_simulacao,valor])
                
                    elif tipo_simulacao == 'AC':
                        amplitude = float(termo[4])
                        phase = float(termo[5])
                        componentes.append(['I',no_a,no_b,tipo_simulacao,amplitude,phase])
            
                        if(max(no_a,no_b) > no_maior):
                            
                            no_maior = max(no_a, no_b)

                    
                
                elif tipo_elemento == 'V':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    tipo_simulacao = termo[3]
                
                    if tipo_simulacao == 'DC':
                        valor = float(termo[4])
                        componentes.append(['V',no_a,no_b,tipo_simulacao,valor])
                
                    elif tipo_simulacao == 'AC':
                        amplitude = float(termo[4])
                        phase = float(termo[5])
                        componentes.append(['V',no_a,no_b,tipo_simulacao,amplitude,phase])
                
                
                        if(max(no_a,no_b) > no_maior):
                            
                            no_maior = max(no_a, no_b)






                elif tipo_elemento == 'E':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    valor = float(termo[5])
                
                    if max(no_a,no_b,no_c,no_d) > no_maior:
                        
                        no_maior = max(no_a,no_b,no_c,no_d)
                    componentes.append(['E',no_a,no_b,no_c,no_d,valor])







                elif tipo_elemento == 'F':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    valor = float(termo[5])
                
                    if max(no_a,no_b,no_c,no_d) > no_maior:
                        
                        no_maior = max(no_a,no_b,no_c,no_d)
                    componentes.append(['F',no_a,no_b,no_c,no_d,valor])
                        
                





                elif tipo_elemento == 'G':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    val = float(termo[5])    
                
                    if(max(no_a,no_b,no_c,no_d) > no_maior):

                        
                        no_maior = max(no_a,no_b,no_c,no_d)
                    componentes.append(['G',no_a,no_b,no_c,no_d,valor])




                
                elif tipo_elemento == 'H':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    valor = float(termo[5])
                
                    if max(no_a,no_b,no_c,no_d) > no_maior:
                        
                        no_maior = max(no_a,no_b,no_c,no_d)
                    componentes.append(['H',no_a,no_b,no_c,no_d,valor])




                elif tipo_elemento == 'A':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    no_e = int(termo[5])
                    no_f = int(termo[6])
                    K1 = float(termo[7])
                    K2 = float(termo[8])
                    K3 = float(termo[9])
                    K4 = float(termo[10])
                    K5 = float(termo[11])
                    K6 = float(termo[12])
                    K7 = float(termo[13])
                    K8 = float(termo[14])
                    I1 = float(termo[15])
                    I2 = float(termo[16])
    
                    if max(no_a, no_b, no_c, no_d, no_e, no_f) > no_maior:
                        no_maior = max(no_a, no_b, no_c, no_d, no_e, no_f)
                    
                    componentes.append(['A', no_a, no_b, no_c, no_d, no_e, no_f, K1, K2, K3, K4, K5, K6, K7, K8, I1, I2])





                elif tipo_elemento == 'B':
                    no_a = int(termo[1])
                    no_b = int(termo[2])
                    no_c = int(termo[3])
                    no_d = int(termo[4])
                    no_e = int(termo[5])
                    K1 = float(termo[6])
                    K2 = float(termo[7])
                    K3 = float(termo[8])
                    K4 = float(termo[9])
                    K5 = float(termo[10])
                    K6 = float(termo[11])

    
                    if max(no_a, no_b, no_c, no_d, no_e) > no_maior:
                        no_maior = max(no_a, no_b, no_c, no_d, no_e,)
                    
                    componentes.append(['A', no_a, no_b, no_c, no_d, no_e, K1, K2, K3, K4, K5, K6])

                

                    
                else:
                    print("Erro. Componente nao encontrado")
                    break

               



    arquivo.close()
    no_maior=no_maior+1
    
    return componentes, no_maior




def grafico (nome_arquivo, tipo_simulacao, lista_nos, parametros, plot):
    freqs,mod,fase = main(nome_arquivo, tipo_simulacao, lista_nos, parametros)

    title = nome_arquivo[54:72]

    for i in range(len(mod)):
        if(plot == 'mod'):
            plt.title(f'Modulo - {title}')
            plt.plot(freqs,mod[i])
            plt.xscale("log")
        if(plot == 'fase'):
            plt.title(f'Fase - {title}')
            plt.plot(freqs,fase[i])
            plt.xscale("log")
    plt.show()   
    return 0


### Main
def main(nome_arquivo, tipo_simulacao, lista_nos, parametros):
    components,num_nos = lerArquivo(nome_arquivo)

    
    if tipo_simulacao == 'DC':


        frequencia=0


        
        A = np.zeros((num_nos,num_nos), dtype=complex)
        vetor_correntes = np.zeros(num_nos)
        
        for elemento in components:
            tipo_componente = elemento[0]

            if tipo_componente == 'R':
                
                A = estampa_Resistor(A,elemento[1],elemento[2],elemento[3])

            elif tipo_componente == 'D':
                
                A = estampa_Diodo(A,elemento[1],elemento[2],elemento[3],elemento[4])
            
            elif tipo_componente == 'G':
                
                A = estampa_FonteG(A,elemento[1],elemento[2],elemento[3],elemento[4],elemento[5])
                
            elif tipo_componente == 'L':
                
                A, vetor_correntes = estampa_Indutor(A, vetor_correntes, elemento[1], elemento[2], elemento[3], frequencia)
                
            elif tipo_componente == 'C':
                
                A = estampa_Capacitor(A, elemento[1], elemento[2], elemento[3], frequencia)
                
            elif tipo_componente == 'K':
                
                A, vetor_correntes = estampa_Transformador(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7], frequencia)
                
            elif tipo_componente == 'F':
                
                A, vetor_correntes = estampa_FonteF(A, vetor_correntes,elemento[1],elemento[2],elemento[3],elemento[4], elemento[5])
                
            elif tipo_componente == 'E':
                
                A, vetor_correntes = estampa_FonteE (A, vetor_correntes,elemento[1],elemento[2],elemento[3],elemento[4], elemento[5])
                
            elif tipo_componente == 'H':
                
                A, vetor_correntes = estampa_FonteH(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5])

                
            elif tipo_componente == 'I':
    
                if elemento[3] == 'DC':
                    vetor_correntes = estampa_FonteI_DC (vetor_correntes,elemento[1],elemento[2],elemento[4])
                elif elemento[3] == 'AC':
                    vetor_correntes = estampa_FonteI_AC (vetor_correntes,elemento[1],elemento[2],0,0)

                                                
            elif tipo_componente == 'V':
                
                if elemento[3] == 'DC':
                    A,vetor_correntes = estampa_FonteV_DC (A,vetor_correntes,elemento[1],elemento[2],elemento[4])
                elif elemento[3] == 'AC':
                    A,vetor_correntes = estampa_FonteV_AC (A,vetor_correntes,elemento[1],elemento[2],0,0)
                    

            elif tipo_componente == 'A':
                A, vetor_correntes = estampa_componenteDC(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7],
                                                          elemento[8], elemento[9], elemento[10], elemento[11], elemento[12], elemento[13], elemento[14], elemento[15], elemento[16])



            elif tipo_componente == 'B':
                A, vetor_correntes = estampa_componenteAC(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7],
                                                          elemento[8], elemento[9], elemento[10], elemento[11])

                        

        A = A[1:,1:]
        
        vetor_correntes = vetor_correntes[1:]
        solucao = np.linalg.solve(A, vetor_correntes)
    
        resposta = []
        for i in range(len(solucao)):
            if i+1 in lista_nos:
                resposta.append(solucao[i])
                
        resposta_str = ', '.join(map(str, resposta))
        resposta_array = np.array([resposta_str])

        
        return np.array(resposta_array)

    elif tipo_simulacao == 'AC':

        frequencia = 2*np.pi*np.logspace( int(np.log10(parametros[0])), int(np.log10(parametros[1])),parametros[2])
        mod = np.zeros((len(lista_nos),frequencia.shape[0]))
        phases = np.zeros((len(lista_nos),frequencia.shape[0]))

        
        for index in range(len(frequencia)):
            A = np.zeros((num_nos,num_nos), dtype=complex)
            vetor_correntes = np.zeros(num_nos, dtype=complex)
            for elemento in components:
                tipo_componente = elemento[0]
                
                if tipo_componente == 'R':
                    
                    A = estampa_Resistor(A,elemento[1],elemento[2],elemento[3])
                    
                elif tipo_componente == 'G':
                    
                    A = estampa_FonteG(A,elemento[1],elemento[2],elemento[3],elemento[4],elemento[5])
                    
                elif tipo_componente == 'L':
                    
                    A, vetor_correntes = estampa_Indutor(A, vetor_correntes, elemento[1], elemento[2], elemento[3], frequencia[index])
                    
                elif tipo_componente == 'C':
                    
                    A = estampa_Capacitor(A, elemento[1], elemento[2], elemento[3], frequencia[index])
                    
                elif tipo_componente == 'K':
                    
                    A, vetor_correntes = estampa_Transformador(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7], frequencia[index])
                    
                elif tipo_componente == 'F':
                    
                    A, vetor_correntes = estampa_FonteF(A, vetor_correntes,elemento[1],elemento[2],elemento[3],elemento[4], elemento[5])
                    
                elif tipo_componente == 'E':
                    
                    A, vetor_correntes = estampa_FonteE(A, vetor_correntes,elemento[1],elemento[2],elemento[3],elemento[4], elemento[5])
                    
                elif tipo_componente == 'H':
                    
                    A, vetor_correntes = estampa_FonteH(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5])
                    
                    
                elif tipo_componente == 'I':
                    
                    if elemento[3] == 'DC':
                        vetor_correntes = estampa_FonteI_DC(vetor_correntes,elemento[1],elemento[2],0)
                    elif elemento[3] == 'AC':
                        vetor_correntes = estampa_FonteI_AC(vetor_correntes,elemento[1],elemento[2],elemento[4],elemento[5])
                        
                        
                elif tipo_componente == 'V':
                    
                    if elemento[3] == 'DC':
                        A, vetor_correntes = estampa_FonteV_DC(A,vetor_correntes,elemento[1],elemento[2],0)
                    elif elemento[3] == 'AC':
                        A, vetor_correntes = estampa_FonteV_AC(A,vetor_correntes,elemento[1],elemento[2],elemento[4],elemento[5])


                elif tipo_componente == 'A':
                    A, vetor_correntes = estampa_componenteDC(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7],
                                                          elemento[8], elemento[9], elemento[10], elemento[11], elemento[12], elemento[13], elemento[14], elemento[15], elemento[16])

                elif tipo_componente == 'B':
                    A, vetor_correntes = estampa_componenteAC(A, vetor_correntes, elemento[1], elemento[2], elemento[3], elemento[4], elemento[5], elemento[6], elemento[7],
                                                          elemento[8], elemento[9], elemento[10], elemento[11])
                
                
            A = A[1:,1:]
            
            vetor_correntes = vetor_correntes[1:]
            solucao = np.linalg.solve(A, vetor_correntes)
            
            for j in range(len(lista_nos)):
                mod[j,index] = 20*np.log10(np.abs(solucao[lista_nos[j]-1]))
                phases[j,index] = np.degrees(np.angle(solucao[lista_nos[j]-1]))

        return frequencia/(2*np.pi),mod,phases

    else:
        return 1






if __name__ == "__main__":
    tensao_nodal = grafico('C:\\Users\\gabri\\OneDrive\\Área de Trabalho\\trabalho2CE2\\./netlistAC10.txt','AC',[4,5], [0.01, 500, 1000], 'mod')
    print(tensao_nodal)
