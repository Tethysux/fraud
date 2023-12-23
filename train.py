import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paso 1: Cargar el conjunto de datos CSV
df = pd.read_csv('credit.csv')

# Paso 2: Explorar y preprocesar los datos
df = df.dropna()

# Normalizar datos numéricos (excluyendo las columnas Time y Class)
scaler = StandardScaler()
df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])

# Paso 3: Separar características y etiquetas
features = df.iloc[:, 1:-1]  # Excluye las columnas Time y Class
labels = df['Class']

# Paso 4: Convertir a tensores de PyTorch
features_tensor = torch.tensor(features.values, dtype=torch.float32)
labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

# Paso 5: Dividir en conjuntos de entrenamiento y prueba
features_train, features_test, labels_train, labels_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, random_state=42)

# Paso 6: Definir la arquitectura de la red neuronal
class MiRedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MiRedNeuronal, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Paso 7: Configurar hiperparámetros, función de pérdida y optimizador
input_size = features_train.shape[1]
hidden_size = 64
output_size = 1

mi_red = MiRedNeuronal(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss para problemas de clasificación binaria
optimizer = optim.Adam(mi_red.parameters(), lr=0.001)

# Paso 8: Entrenar la red neuronal
num_epochs = 900

for epoch in range(num_epochs):
    # Forward pass
    outputs = mi_red(features_train)
    loss = criterion(outputs, labels_train.view(-1, 1))

    # Backward pass y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Imprimir la pérdida cada 10 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Paso 9: Evaluar el modelo en el conjunto de prueba
mi_red.eval()  # Cambiar a modo evaluación

with torch.no_grad():
    outputs_test = mi_red(features_test)
    predictions = torch.sigmoid(outputs_test) >= 0.5

accuracy = torch.sum(predictions == labels_test.view(-1, 1)).item() / len(labels_test)
print(f'Accuracy on Test Set: {accuracy:.4f}')

# Calcular métricas adicionales
precision = precision_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
f1 = f1_score(labels_test, predictions)

# Matriz de Confusión
conf_matrix = confusion_matrix(labels_test, predictions)

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
# Guardar el modelo entrenado
torch.save(mi_red.state_dict(), 'trained.pth')
