import matplotlib.pyplot as plt
import numpy as np

# Datos
models = ['Logistic Regression', 'Transformer (main)']
f1_changes = [0.4528, 0.7078]
avg_f1 = [0.28, 0.5174]  # avg_f1 para regresión logística es estimado ~0.28

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, f1_changes, width, label='F1 (Changes)', color='steelblue')
bars2 = ax.bar(x + width/2, avg_f1, width, label='Avg F1 (Per-Network)', color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Comparison: Logistic Regression vs Transformer', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Añadir valores encima de las barras
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('bar_chart_comparison_logreg_vs_transformer.png', dpi=300)
plt.show()
print("✅ Gráfico guardado como bar_chart_comparison_logreg_vs_transformer.png")