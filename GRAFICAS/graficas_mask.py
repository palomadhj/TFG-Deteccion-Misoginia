import matplotlib.pyplot as plt
import pandas as pd


def poner_datos_bien(subplot, data, nombre):
    x = data['Step']
    y = data['Value']
    subplot.plot(x,y, label=nombre)

if __name__ == "__main__":
    data_spacy1 = pd.read_csv('GPT_EXP4_train (1).csv')
    data_spacy2 = pd.read_csv('GPT2_EXP4_train (1).csv')
    data_spacy3 = pd.read_csv('XLNET2_train (1).csv')
    data2_spacy1 = pd.read_csv('accuracy_intent/GPT_EXP4_train.csv')
    data2_spacy2 = pd.read_csv('accuracy_intent/GPT2_EXP4_train.csv')
    data2_spacy3 = pd.read_csv('accuracy_intent/XLNET2_train.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
    plt.title('Accuracy')
    poner_datos_bien(ax1, data_spacy1, nombre = 'GPT')
    poner_datos_bien(ax1, data_spacy2, nombre = 'GPT 2')
    poner_datos_bien(ax1, data_spacy3, nombre = 'XLNet')
    poner_datos_bien(ax2, data2_spacy1, nombre = 'GPT')
    poner_datos_bien(ax2, data2_spacy2, nombre = 'GPT 2')
    poner_datos_bien(ax2, data2_spacy3, nombre = 'XLNet')
    #plt.tight_layout()
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax1.set_title('Mask accuracy')
    ax2.set_title('Intent accuracy')
    # Personaliza las etiquetas de los ejes (axes) si lo deseas
    ax1.grid()
    ax2.grid()
    plt.show()


"""import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV con Pandas
data = pd.read_csv('datos.csv')

# Obtener las columnas del archivo CSV
x = data['Columna_X']
y1 = data['Columna_Y1']
y2 = data['Columna_Y2']

# Crear una figura y dos subgráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Graficar en el primer subgráfico
ax1.plot(x, y1)
ax1.set_xlabel('Eje X')
ax1.set_ylabel('Eje Y1')
ax1.set_title('Gráfico 1')

# Graficar en el segundo subgráfico
ax2.plot(x, y2)
ax2.set_xlabel('Eje X')
ax2.set_ylabel('Eje Y2')
ax2.set_title('Gráfico 2')

# Ajustar los espacios entre subgráficos
plt.tight_layout()

# Mostrar la figura con ambos subgráficos
plt.show()"""