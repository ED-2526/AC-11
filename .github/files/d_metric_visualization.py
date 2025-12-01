import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Leer el JSON
json_path = os.path.join(os.path.dirname(__file__), 'estadisticas_k2.json')
with open(json_path, 'r') as f:
    data = json.load(f)

# Extraer valores de K y ordenarlos
k_values = sorted([int(k) for k in data.keys()])

# Extraer métricas principales
metrics = {
    'accuracy': [],
    'precision_macro': [],
    'precision_micro': [],
    'precision_weighted': [],
    'recall_macro': [],
    'recall_micro': [],
    'recall_weighted': [],
    'f1_macro': [],
    'f1_micro': [],
    'f1_weighted': []
}

for k in k_values:
    k_str = str(k)
    for metric in metrics.keys():
        metrics[metric].append(data[k_str][metric])

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Evaluación de Métricas vs Tamaño del Vocabulario (K)', fontsize=16, fontweight='bold')

# 1. Accuracy
axes[0, 0].plot(k_values, metrics['accuracy'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
axes[0, 0].set_xlabel('Tamaño del Vocabulario (K)', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Accuracy vs K', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xscale('log')

# Encontrar el mejor K para accuracy
best_k_acc = k_values[np.argmax(metrics['accuracy'])]
best_acc = max(metrics['accuracy'])
axes[0, 0].axvline(x=best_k_acc, color='red', linestyle='--', alpha=0.5, label=f'Mejor: K={best_k_acc} (Acc={best_acc:.3f})')
axes[0, 0].legend()

# 2. Precision (macro, micro, weighted)
axes[0, 1].plot(k_values, metrics['precision_macro'], 'o-', label='Macro', linewidth=2, markersize=6)
axes[0, 1].plot(k_values, metrics['precision_micro'], 's-', label='Micro', linewidth=2, markersize=6)
axes[0, 1].plot(k_values, metrics['precision_weighted'], '^-', label='Weighted', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Tamaño del Vocabulario (K)', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision vs K', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].set_xscale('log')

# 3. Recall (macro, micro, weighted)
axes[1, 0].plot(k_values, metrics['recall_macro'], 'o-', label='Macro', linewidth=2, markersize=6)
axes[1, 0].plot(k_values, metrics['recall_micro'], 's-', label='Micro', linewidth=2, markersize=6)
axes[1, 0].plot(k_values, metrics['recall_weighted'], '^-', label='Weighted', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Tamaño del Vocabulario (K)', fontsize=11)
axes[1, 0].set_ylabel('Recall', fontsize=11)
axes[1, 0].set_title('Recall vs K', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].set_xscale('log')

# 4. F1-Score (macro, micro, weighted)
axes[1, 1].plot(k_values, metrics['f1_macro'], 'o-', label='Macro', linewidth=2, markersize=6)
axes[1, 1].plot(k_values, metrics['f1_micro'], 's-', label='Micro', linewidth=2, markersize=6)
axes[1, 1].plot(k_values, metrics['f1_weighted'], '^-', label='Weighted', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Tamaño del Vocabulario (K)', fontsize=11)
axes[1, 1].set_ylabel('F1-Score', fontsize=11)
axes[1, 1].set_title('F1-Score vs K', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].set_xscale('log')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'metrics3.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path}")

# Crear un segundo gráfico con métricas por clase para el mejor K
best_k_str = str(best_k_acc)
clases = ['burger', 'butter_naan', 'chai', 'chapati']

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(f'Métricas por Clase (K={best_k_acc})', fontsize=16, fontweight='bold')

# Extraer métricas por clase
class_metrics = data[best_k_str]['metricas_por_clase']
precision_per_class = [class_metrics[cls]['precision'] for cls in clases]
recall_per_class = [class_metrics[cls]['recall'] for cls in clases]
f1_per_class = [class_metrics[cls]['f1-score'] for cls in clases]

x = np.arange(len(clases))
width = 0.25

# Gráfico de barras
axes2[0].bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
axes2[0].bar(x, recall_per_class, width, label='Recall', alpha=0.8)
axes2[0].bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
axes2[0].set_xlabel('Clase', fontsize=11)
axes2[0].set_ylabel('Score', fontsize=11)
axes2[0].set_title('Métricas por Clase', fontsize=12, fontweight='bold')
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(clases, rotation=45, ha='right')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3, axis='y')

# Matriz de confusión
conf_matrix = np.array(data[best_k_str]['confusion_matrix'])
im = axes2[1].imshow(conf_matrix, cmap='Blues', aspect='auto')
axes2[1].set_xticks(np.arange(len(clases)))
axes2[1].set_yticks(np.arange(len(clases)))
axes2[1].set_xticklabels(clases, rotation=45, ha='right')
axes2[1].set_yticklabels(clases)
axes2[1].set_xlabel('Predicción', fontsize=11)
axes2[1].set_ylabel('Real', fontsize=11)
axes2[1].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')

# Añadir valores en la matriz
for i in range(len(clases)):
    for j in range(len(clases)):
        text = axes2[1].text(j, i, conf_matrix[i, j],
                           ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max()/2 else "white",
                           fontsize=10, fontweight='bold')

plt.colorbar(im, ax=axes2[1])
plt.tight_layout()
output_path2 = os.path.join(os.path.dirname(__file__), 'metrics4.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {output_path2}")

# Imprimir resumen
print(f"\n=== RESUMEN ===")
print(f"Mejor K: {best_k_acc}")
print(f"Accuracy máxima: {best_acc:.4f}")
print(f"Rango de K evaluado: {min(k_values)} - {max(k_values)}")