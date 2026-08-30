import pickle
import matplotlib.pyplot as plt
import numpy as np

# ------------------- Datos de la configuración principal (ya entrenada) -------------------
# Si no tienes los .pkl de la principal, pon los números a mano:
main_config = {
    'name': 'Main',
    'val_losses': [4.8763, 1.5494, 1.4986, 1.4838, 1.4777, 1.4391, 1.4450, 1.4055, 1.4157, 1.4206, 1.4170, 1.4370, 1.4309, 1.4099, 1.4086, 1.4091, 1.4035, 1.4023, 1.3975, 1.4034, 1.3940, 1.4155, 1.4001, 1.4011, 1.3960, 1.3937, 1.4065, 1.3723, 1.4098, 1.3968],
    'steps': list(range(0, 3000, 100)),
    'f1_changes': 0.7078,
    'avg_f1': 0.5174,
    'config': {'dropout': 0.0, 'n_layer': 2, 'n_embd': 64}
}

# Cargar las configuraciones A, B, C desde los archivos .pkl
configs = {}
for c in ['A_dropout', 'B_deeper', 'C_deeper_dropout']:
    try:
        with open(f'results_{c}.pkl', 'rb') as f:
            res = pickle.load(f)
            configs[c] = {
                'name': c.replace('_', ' '),
                'val_losses': res['val_losses'],
                'steps': res['steps'],
                'f1_changes': res['f1_changes'],
                'avg_f1': sum(res['f1_per_network']) / 7,
                'config': res['config']
            }
    except FileNotFoundError:
        print(f"⚠️ No se encontró results_{c}.pkl")

# Añadir la principal
all_configs = {'Main': main_config, **configs}

# ------------------- Gráfica 1: Curva de pérdidas -------------------
plt.figure(figsize=(10, 6))
colors = ['black', 'blue', 'green', 'red']
markers = ['o', 's', '^', 'D']
for i, (name, cfg) in enumerate(all_configs.items()):
    if len(cfg['steps']) == len(cfg['val_losses']):
        plt.plot(cfg['steps'], cfg['val_losses'], 
                 label=name, color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)], markersize=4, linewidth=2)
    else:
        # Si el número de puntos no coincide (por si acaso)
        steps = list(range(0, 3000, 100))
        plt.plot(steps[:len(cfg['val_losses'])], cfg['val_losses'], 
                 label=name, color=colors[i % len(colors)], linewidth=2)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('Validation Loss Comparison Across Configurations', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curves_comparison.png', dpi=300)
plt.show()
print("✅ Gráfica de pérdidas guardada como loss_curves_comparison.png")

# ------------------- Gráfica 2: Barras comparativas -------------------
names = list(all_configs.keys())
f1_changes = [all_configs[n]['f1_changes'] for n in names]
avg_f1 = [all_configs[n]['avg_f1'] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_changes, width, label='F1 (Changes)', color='steelblue')
bars2 = ax.bar(x + width/2, avg_f1, width, label='Avg F1 (Per-Network)', color='coral')

ax.set_xlabel('Configuration', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Comparison of Transformer Configurations', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Añadir valores encima de las barras
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('bar_chart_comparison.png', dpi=300)
plt.show()
print("✅ Gráfico de barras guardado como bar_chart_comparison.png")