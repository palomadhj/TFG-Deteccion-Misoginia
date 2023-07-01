import matplotlib.pyplot as plt
import numpy as np

# Definir la matriz de confusión
confusion_matrix = np.array([[225, 69],
                             [85, 272]])

# Obtener las etiquetas de las clases
labels = ["misos", "no_misos"]

# Crear la figura y el eje
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap='Blues')

# Agregar título y etiquetas de los ejes
ax.set_title("Matriz de Confusión")
ax.set_xlabel("Etiqueta Predicha")
ax.set_ylabel("Etiqueta Real")
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Mostrar los valores en cada celda
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center", color="black")

# Mostrar la barra de color para los valores
cbar = ax.figure.colorbar(im, ax=ax)

# Ajustar la ubicación de los elementos en la figura
plt.tight_layout()

# Mostrar la figura
plt.show()