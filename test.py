# =====================================================
# TRABAJO FINAL - CLASIFICACIÓN DE DÍGITOS MANUSCRITOS
# TecnoForms
# =====================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("=== SOLUCIÓN COMPLETA PARA TECNOFORMS ===\n")

# Carga de datos
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocesamiento
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("Preprocesamiento: Normalización + Aplanado completado")

# ==================== ALGORITMOS DE COMPARACIÓN ====================
print("\n--- Comparación de Algoritmos ---")

# 1. Regresión Logística
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_flat[:8000], y_train[:8000])
print("Precisión Regresión Logística:", accuracy_score(y_test[:2000], logreg.predict(X_test_flat[:2000])))

# 2. K-Means (no supervisado)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train_flat[:5000])

# 3. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_flat[:10000], y_train[:10000])
print("Precisión KNN:", accuracy_score(y_test[:2000], knn.predict(X_test_flat[:2000])))

# ==================== CNN (Modelo Principal) ====================
print("\n--- Red Neuronal Convolucional (CNN) ---")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3,3), activation='relu'),     # Capa convolucional
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train.reshape(-1,28,28,1), y_train, 
                    epochs=8, batch_size=128, validation_split=0.2, verbose=1)

# Evaluación
y_pred = np.argmax(model.predict(X_test.reshape(-1,28,28,1)), axis=1)

print(f"\nPrecisión Final CNN: {accuracy_score(y_test, y_pred):.4f}")
print("\nMétricas detalladas:")
print(classification_report(y_test, y_pred))

# Matriz de Confusión
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - CNN')
plt.show()

# Guardar modelo
model.save('mnist_model.h5')
print("\nModelo guardado correctamente.")

print("\n¡Código mejorado y listo para el informe!")