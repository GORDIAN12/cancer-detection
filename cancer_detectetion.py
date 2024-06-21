import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from scipy.stats import norm, rv_histogram


datos = pd.read_csv('p.csv')
X=pd.read_excel("PERIMETER.xlsx")
Y=pd.read_csv('TEXTURA.csv')
benignos=Y[Y['diagnosis']=='B']
benignos_T=benignos['texture']
malignos=Y[Y['diagnosis']=='M']
malignos_T=malignos['texture']

two_texture = datos["textura"]

min_perimeter, max_perimeter = min(two_texture), max(two_texture)
n_bins = math.ceil((max_perimeter - min_perimeter))

hist, bin_edges = np.histogram(two_texture, bins=n_bins, density=True)
hist_dis = stats.rv_histogram(histogram=(hist, bin_edges))

meanz = np.mean(two_texture)
stdz = np.std(two_texture)
n_z = len(two_texture)
se = stdz / np.sqrt(n_z)

#HACIENDO USO DEL METODO STUDENT
tstar = 2.064
zstar = 1.96
lcb = meanz - tstar * se
ucb = meanz + tstar * se

# Crear gráfico de histograma y PDF
st.header("Histograma y Distribución de Datos de Cancer")
fig, ax = plt.subplots()
_, _, patches = ax.hist(two_texture, bins=n_bins, edgecolor='white', density=True, label="Histograma")
eje_x = np.linspace(min_perimeter, max_perimeter, n_bins)
ax.plot(eje_x, hist_dis.pdf(eje_x), label="PDF", color='orange')

media_estimada = np.mean(two_texture)
desviacion_estimada = np.std(two_texture)

x_range = np.linspace(np.min(two_texture) - 1, np.max(two_texture) + 1)
y_normal = norm.pdf(x_range, loc=media_estimada, scale=desviacion_estimada)
ax.plot(x_range, y_normal, 'r', label="Distribución normal ajustada")

ax.set_title("Histograma de Textura de Tumores")
ax.set_xlabel("Textura")
ax.set_ylabel("Densidad")
ax.legend()

st.header("BENIGNOS Y MALIGNOS")
mediaB=f"Media : {meanz:.3f}"
ax.text(1, 0.95, mediaB, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
desviacionB=f"Desviación de texturas: {stdz:.3f}"
ax.text(1, 0.85, desviacionB, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
errorB=f"El error estándar es: {se:.3f}"
ax.text(1, 0.75, errorB, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

interB=f"Intervalo de confianza es =  ({lcb:.3f}, {ucb:.3f})"
ax.text(1, 0.65, interB, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

# Mostrar gráfico en el dashboard
st.pyplot(fig)

#DATOS DE BENIGNOS y malignos
textureB=benignos_T
textureM=malignos_T

min_textureB, max_textureB=min(textureB),max(textureB)
n_binsB=math.ceil((max_textureB - min_textureB))

min_textureM, max_textureM=min(textureM),max(textureM)
n_binsM=math.ceil((max_textureM - min_textureM))

st.title("Textura de Tumores Benignos y Malignos")

# Crear gráfico de histogramas
st.header("Histograma de Texturas")

fig, ax = plt.subplots()

ax.hist(textureB, bins=n_binsB, alpha=0.5, label='Benignos', edgecolor='white', color='blue')
ax.hist(textureM, bins=n_binsM, alpha=0.5, label='Malignos', edgecolor='white', color='orange')

ax.legend(bbox_to_anchor=(1.05, 1))
ax.set_title("Histograma de Texturas de Tumores Benignos y Malignos")
ax.set_xlabel("Textura")
ax.set_ylabel("Frecuencia")

# Mostrar gráfico en el dashboard
st.pyplot(fig)

media_estimada = np.mean(textureB)
desviacion_estimada = np.std(textureB)


x_range = np.linspace(np.min(textureB) - 1, np.max(textureB) + 1)

y_normal = norm.pdf(x_range, loc=media_estimada, scale=desviacion_estimada)

# Calcular el histograma y convertirlo en una distribución
hist, bin_edges = np.histogram(textureB, bins=n_binsB, density=True)
hist_dis = rv_histogram((hist, bin_edges))

two_texture=Y["texture"]
meanz=np.mean(two_texture)
stdz=np.std(two_texture)
se=stdz/np.sqrt(n_z)

##############HISTOGRAMA DE BENIGNOS SOLAMENTE#############

# Crear gráfico de histograma y PDF
st.header("Histograma y Distribución de Textura")
fig, ax = plt.subplots()

# Histograma
ax.hist(textureB, bins=n_binsB, edgecolor='white', density=True, label="Histograma")

# PDF estimada
eje_x = np.linspace(np.min(textureB), np.max(textureB), n_binsB)
ax.plot(eje_x, hist_dis.pdf(eje_x), label="PDF", color='orange')

# Distribución normal ajustada
x_range = np.linspace(np.min(textureB) - 1, np.max(textureB) + 1)
y_normal = norm.pdf(x_range, loc=media_estimada, scale=desviacion_estimada)
ax.plot(x_range, y_normal, 'r', label="Distribución normal ajustada")


meanz2=np.mean(benignos_T)
stdz2=np.std(benignos_T)
n_z2=len(benignos_T)
se2=stdz/np.sqrt(n_z)
tstar2=2.064
zstar=1.96
lcb2=meanz2-tstar2*se2
ucb2=meanz2+tstar2*se2
# Añadir intervalo de confianza al gráfico

st.header("Estadísticas de Textura Benignos")
mediaBenig=f"Media estimada: {media_estimada:.3f}"
ax.text(1, 0.95, mediaBenig, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')
desviacionBenig=f"Desviación estándar estimada: {desviacion_estimada:.3f}"
ax.text(1, 0.85, desviacionBenig, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

intervalo_confianza_str = f"Intervalo confianza: ({lcb2:.3f}, {ucb2:.3f})"
ax.text(1, 0.75, intervalo_confianza_str, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

# Añadir título y etiquetas
ax.set_title("Histograma de Textura de Tumores Benignos")
ax.set_xlabel("Textura")
ax.set_ylabel("Densidad")
ax.legend()

st.pyplot(fig)


############### iteracion de graficos ###########################
benig=benignos_T
# Tamaños de muestra y número de muestras
tamano_muestra = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
num_muestras = 3000

# Configurar Streamlit
st.title("Histogramas de Promedios por Tamaño de Muestra")

# Bucle para generar y mostrar los histogramas en Streamlit
for tamano in tamano_muestra:
        fig, ax = plt.subplots()
        promedios = np.array([])
        for i in range(num_muestras):
                muestra = np.random.choice(benig, tamano)
                promedios = np.append(promedios, muestra.mean())

        # Graficar histograma
        ax.hist(promedios, bins='auto', alpha=0.7, edgecolor="white")
        ax.set_title(f"Tamaño de muestra = {tamano}")
        ax.set_xlabel("Promedio")
        ax.set_ylabel("Frecuencia")

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)


######################### HISTOGRAMA DE MALIGNOS

st.header("Histograma y Distribución de Textura de Malignos")
fig, ax = plt.subplots()
media_estimada = np.mean(textureM)
desviacion_estimada = np.std(textureM)


# Definir el rango para el gráfico
x_range = np.linspace(np.min(textureM) - 1, np.max(textureM) + 1, 500)

# Calcular la PDF normal ajustada
y_normal = norm.pdf(x_range, loc=media_estimada, scale=desviacion_estimada)

####################DATOS MALIGNOS E HISTOGRAMA###########
hist=np.histogram(textureM, bins=n_binsM)
hist_dis=stats.rv_histogram(histogram=hist, density=True)

ax.hist(textureM, bins=n_binsM, edgecolor='white' , density=True, label="Histograma")
eje_x=np.linspace(min_textureM, max_textureM, n_binsM)
ax.plot(eje_x, hist_dis.pdf(eje_x), label="PDF")

ax.plot(x_range, y_normal, 'r', label="Distribución normal ajustada")


ax.set_title("Hsitograma de textura Malignos")
ax.set_xlabel("Longitud de textura")
ax.set_ylabel("numero de ocurrencias")
#plt.bar_label(z)
ax.legend()

print(f"Probabilidad de que perimetro = 20:  {hist_dis.pdf(20):.3f}")
meanz3=np.mean(malignos_T)
stdz3=np.std(malignos_T)
n_z3=len(malignos_T)
mediaMalignos=f"media de malignos perimetro es:  {meanz3:.3f}"
ax.text(1, 0.95, mediaMalignos, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

varianzaMalignos=f"Varianza malignos perimetro es:  {stdz3:.3f}"
ax.text(1, 0.85, varianzaMalignos, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

se3=stdz3/np.sqrt(n_z3)
error3=f"EL error estandar es: {se3:.3f}"
ax.text(1, 0.75,  error3, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

tstar=2.064
zstar=1.96
lcb3=meanz3-tstar*se3
ucb3=meanz3+tstar*se3
intervalo_confianza3=f"Intervalo confianza: ({lcb3:.3f}, {ucb3:.3f})"
ax.text(1, 0.65,  intervalo_confianza3, transform=ax.transAxes, fontsize=12,
        verticalalignment='top')

st.pyplot(fig)

############### iteracion de graficos ###########################
malig=malignos_T
# Tamaños de muestra y número de muestras
tamano_muestra = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
num_muestras = 3000

# Configurar Streamlit
st.title("Histogramas por tamaño de muestra Malignos")

# Bucle para generar y mostrar los histogramas en Streamlit
for tamano in tamano_muestra:
        fig, ax = plt.subplots()
        promedios = np.array([])
        for i in range(num_muestras):
                muestra = np.random.choice(malig, tamano)
                promedios = np.append(promedios, muestra.mean())

        # Graficar histograma
        ax.hist(promedios, bins='auto', alpha=0.7, edgecolor="white")
        ax.set_title(f"Tamaño de muestra = {tamano}")
        ax.set_xlabel("Promedio")
        ax.set_ylabel("Frecuencia")

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

st.header('Histogramas 3D')
# Leer archivo CSV
datos = pd.read_csv('p.csv')

# Función para obtener los datos de la característica
def get_feature_data(var):
    if var in datos.columns:
        return datos[[var, 'diagnosis']]
    raise ValueError(f"Variable '{var}' no encontrada en los datos")

# Función para filtrar por diagnóstico
def filter_by_diagnosis(feature, diagnosis, var):
    if diagnosis in ['B', 'M']:
        return feature[feature['diagnosis'] == diagnosis][var]
    return feature[var]

# Función principal para actualizar el gráfico
def update_graph(var1, var2, numBins1, numBins2, diag):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    feature1 = get_feature_data(var1)
    feature2 = get_feature_data(var2)

    feature1 = filter_by_diagnosis(feature1, diag, var1)
    feature2 = filter_by_diagnosis(feature2, diag, var2)

    label_prefix = {'B': 'CANCER BENIGNO', 'M': 'CANCER MALIGNO', 'Mezcla': 'Mezcla'}.get(diag, 'Mezcla')

    hist, xedges, yedges = np.histogram2d(feature1, feature2, bins=[numBins1, numBins2], density=True)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    xpos, ypos = np.meshgrid(xcenters, ycenters, indexing="ij")

    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
    dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
    dz = hist.ravel()

    # Definir la paleta de colores según el diagnóstico
    if diag == 'B':
        cmap = plt.cm.BrBG
    elif diag == 'M':
        cmap = plt.cm.Wistia
    else:
        cmap = plt.cm.plasma

    rgba = [cmap((k - np.min(dz)) / (np.max(dz) - np.min(dz))) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.9, color=rgba, linewidth=0.5, edgecolor='white')

    mean = [feature1.mean(), feature2.mean()]
    cov = np.cov(feature1, feature2)
    x = np.linspace(feature1.min(), feature1.max(), 200)
    y = np.linspace(feature2.min(), feature2.max(), 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    normal_b = stats.multivariate_normal(mean, cov)
    Z = normal_b.pdf(pos)

    plot = ax.plot_surface(X, Y, Z, linewidth=0.1, rstride=5, cstride=5, edgecolor='black', cmap=cmap, alpha=0.3)
    fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10, pad=-0.85)

    ax.set_title(f'Diagnóstico: {label_prefix}')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel("Frecuencia")

    textstr = f'variable 1 = {feature1.mean():.3f}\nvariable 2 = {feature2.mean():.3f}'
    ax.text2D(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top')

    #print(f'covx: {math.sqrt(cov[0][0]):.3f}')
    #print(f'covy: {math.sqrt(cov[1][1]):.3f}')
    st.pyplot(fig)


# Llamada a la función con parámetros de ejemplo
update_graph('textura', 'perimeter', 15, 15, 'B')
update_graph('textura', 'radius', 20, 20, 'M')
