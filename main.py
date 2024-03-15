import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class ACP:
    def __init__(self, datos, n_componentes=5):
        self.datos_old = datos
        self.__datos = datos
        self.datos_scaled = None
        self.__n_componentes = n_componentes
        self.matriz_correlaciones = None
        self.valores_propios = None
        self.vectores_propios = None
        self.matriz_de_vectores_propios = None
        self.matriz_componentes = None
        self.matriz_calidades = None
        self.coordenadas = None
        self.calidades_variables = None
        self.vector_inercias = None

    @property
    def datos(self):
        return self.__datos
    
    @datos.setter
    def datos(self, datos):
        self.__datos = datos

    def centrar_reducir_datos(self):
        media = np.mean(self.__datos, axis=0)
        desviacion_estandar = np.std(self.__datos, axis=0)
        matriz_centrada_reducida = (self.__datos - media) / desviacion_estandar
        self.datos_scaled = matriz_centrada_reducida


    def calcular_matriz_correlaciones(self):
        #datos_traspuesta = np.transpose(self.__datos)
        #n = self.__datos.shape[0]
        #inversa_n = 1 / n
        #multply_datos_by_datos_traspuesta = np.matmul(datos_traspuesta, self.matriz_correlaciones)
        #self.matriz_correlaciones = inversa_n * multply_datos_by_datos_traspuesta
        #self.__datos = np.corrcoef(self.__datos, rowvar=False)
        df = pd.DataFrame(self.datos_scaled)
        self.matriz_correlaciones = df.corr()

    def calcular_autovectores_autovalores(self):
        self.valores_propios, self.vectores_propios = np.linalg.eig(self.matriz_correlaciones)


        orden_valores_propios = np.argsort(self.valores_propios)[::-1]
        self.valores_propios = self.valores_propios[orden_valores_propios]
        self.vectores_propios = self.vectores_propios[:,orden_valores_propios]


    def equis_por_correlaciones(self):
        self.matriz_componentes = np.matmul(self.datos_scaled, self.vectores_propios)
        self.matriz_componentes = np.array(self.matriz_componentes)
        for i in range(len(self.matriz_componentes)):
            self.matriz_componentes[i][2] *= -1
        self.matriz_componentes= pd.DataFrame(self.matriz_componentes)

        
    
    def calcualar_matriz_calidades(self):
        ref = self.datos_scaled * self.datos_scaled
        suma_filas_datos_old = np.sum(ref, axis=1)
        suma_filas_datos_old = suma_filas_datos_old.values  # Convert to NumPy array
        self.matriz_calidades = (self.matriz_componentes * self.matriz_componentes) / suma_filas_datos_old[:, np.newaxis]


    def calcular_coordenadas(self):
        # refM = np.vstack(self.vectores_propios)
        # self.coordenadas = np.dot(self.valores_propios, refM)
        matriz = np.zeros((len(self.vectores_propios), len(self.valores_propios)))

        for i, vector in enumerate(self.vectores_propios):
            matriz[i] = np.sqrt(self.valores_propios) * vector

        self.coordenadas = matriz

    def calcular_matriz_calidades_variables(self):
        # refM = np.vstack(self.vectores_propios)
        # self.coordenadas = np.dot(self.valores_propios, refM)
        matriz = np.zeros((len(self.vectores_propios), len(self.valores_propios)))
        
        for i, vector in enumerate(self.vectores_propios):
            matriz[i] = vector * vector
            matriz[i] = matriz[i] * self.valores_propios

        np.set_printoptions(suppress=True, precision=4)

        self.calidades_variables = matriz
        self.calidades_variables= pd.DataFrame(self.calidades_variables)

    def calcular_vector_inercias(self):
        self.vector_inercias = 100 * self.valores_propios / self.matriz_correlaciones.shape[1]

    def plot (self):
        componentes = np.array(self.matriz_componentes)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi=200)
        plt.axhline(0, color='black', linestyle='dashed') 
        plt.axvline(0, color='black', linestyle='dashed')  
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        for i in range (componentes.shape[0]):
            x = componentes[i, 0]
            y = componentes[i, 1]
            ax.scatter(x,y)
            ax.annotate(text= self.datos_scaled.index[i] , xy=(x+0.05,y))
        plt.grid(True)
        plt.show()

    def plot_circulo(self, ejes=[0,1], var_labels = True, titulo = 'Circulo de Correlación'):
        varexp = [x * 100 for x in self.vector_inercias]
        cor = self.matriz_correlaciones.iloc[:, ejes].values
        #plt.style.use('seaborn-whitegrid')
        c = plt.Circle((0,0), radius = 1, color = 'steelblue', fill= False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(0, color='black', linestyle='dashed') 
        plt.axvline(0, color='black', linestyle='dashed') 

        inercia_x= round((varexp[ejes[0]])/100,2)
        inercia_y= round((varexp[ejes[1]])/100,2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')


        for i in range(cor.shape[0]):
            plt.arrow(0,0, cor[i,0] * 0.95, cor[i,1] * 0.95, color = 'steelblue', alpha = 0.5, head_width= 0.05, head_length = 0.05)
        #if var_labels
        plt.grid(True)
        plt.show()

in_datos = pd.read_csv('EjemploEstudiantes.csv', delimiter=';', decimal=",", index_col=0)

acp = ACP(datos=in_datos)
acp.centrar_reducir_datos()
acp.calcular_matriz_correlaciones()
acp.calcular_autovectores_autovalores()
acp.equis_por_correlaciones()
acp.calcualar_matriz_calidades()
acp.calcular_coordenadas()
acp.calcular_matriz_calidades_variables()
acp.calcular_vector_inercias()
print('Matriz de componentes principales\n', acp.matriz_componentes)
print('Matriz de calidades de los individuos\n', acp.matriz_calidades)
print('Matriz de calidades de las variables\n', acp.calidades_variables)
print('Vector de inercias de los ejes\n', acp.vector_inercias)
acp.plot()
acp.plot_circulo()
