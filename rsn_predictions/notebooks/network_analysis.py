import pickle
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- Cargar datos -------------------
with open('state_codes_all.pkl', 'rb') as f:
    codes = pickle.load(f)

print(f"Total tokens: {len(codes)}")

# Función para convertir token a bits
def token_to_bits(token):
    return [(token >> b) & 1 for b in range(6, -1, -1)]

# Nombres de las redes
network_names = ['Visual', 'Somatomotor', 'Dorsal Attention', 
                 'Salience/Ventral Attention', 'Limbic', 'Control', 'Default']

# ------------------- 1. Reglas de asociación -------------------
print("\n🔍 Calculando reglas de asociación...")

# Dividir en runs de 256 (asumimos 90 sujetos * 3 runs)
run_length = 256
n_runs = len(codes) // run_length
print(f"Runs encontrados: {n_runs}")

all_rules = []

for run_idx in range(n_runs):
    start = run_idx * run_length
    end = start + run_length
    run_codes = codes[start:end]
    
    # Convertir a binario
    run_binary = np.array([token_to_bits(t) for t in run_codes])
    
    # Para cada red objetivo (1..7), buscar transiciones donde la red pasa de 0 a 1
    for target_net in range(7):
        # Estado actual (t) y siguiente (t+1)
        current = run_binary[:-1]
        next_state = run_binary[1:]
        
        # Transacciones donde la red target se activa (0→1)
        activation_transitions = (current[:, target_net] == 0) & (next_state[:, target_net] == 1)
        
        # Si no hay activaciones, saltamos
        if np.sum(activation_transitions) < 3:
            continue
        
        # Para cada transición de activación, guardamos las redes activas en t
        transactions = []
        for idx in np.where(activation_transitions)[0]:
            active_networks = [i for i in range(7) if current[idx, i] == 1]
            if active_networks:
                transactions.append(active_networks)
        
        if len(transactions) < 2:
            continue
        
        # Aplicar Apriori
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        try:
            frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
            if len(frequent_itemsets) == 0:
                continue
            
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            if len(rules) == 0:
                continue
            
            # Filtrar reglas que contengan la red target en el consecuente
            for _, rule in rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                # Si el consecuente incluye la red target (y no incluye otras)
                if len(consequents) == 1 and consequents[0] == f'net_{target_net}':
                    all_rules.append({
                        'run': run_idx,
                        'target_net': target_net,
                        'antecedents': antecedents,
                        'support': rule['support'],
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
        except:
            continue

# Agrupar reglas similares
if all_rules:
    rules_df = pd.DataFrame(all_rules)
    # Agrupar por combinación de antecedentes y red objetivo, promediar lift
    rules_grouped = rules_df.groupby(['target_net', 'antecedents']).agg({
        'support': 'mean',
        'confidence': 'mean',
        'lift': 'mean'
    }).reset_index()
    
    # Top 10 reglas por lift
    top_rules = rules_grouped.sort_values('lift', ascending=False).head(10)
    print("\n🏆 Top 10 reglas de asociación (por lift):")
    print(top_rules.to_string(index=False))
else:
    print("⚠️ No se encontraron reglas significativas. Prueba a reducir min_support.")

# Guardar top rules
if 'top_rules' in locals():
    top_rules.to_csv('top_association_rules.csv', index=False)
    print("\n✅ Reglas guardadas en top_association_rules.csv")

# ------------------- 2. Correlaciones cruzadas con retardo -------------------
print("\n📊 Calculando matrices de correlación cruzada...")

# Calcular correlaciones para cada run y promediar
corr_lag1 = np.zeros((7, 7))
corr_lag2 = np.zeros((7, 7))
count = 0

for run_idx in range(n_runs):
    start = run_idx * run_length
    end = start + run_length
    run_codes = codes[start:end]
    run_binary = np.array([token_to_bits(t) for t in run_codes])
    
    # Correlación con retardo 1: red i en t vs red j en t+1
    for i in range(7):
        for j in range(7):
            if i == j:
                continue
            x = run_binary[:-1, i]
            y = run_binary[1:, j]
            if np.std(x) > 0 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isnan(corr):
                    corr_lag1[i, j] += corr
    count += 1

# Promediar
corr_lag1 /= count
corr_lag2 /= count

# Guardar matrices
np.savetxt('corr_lag1.csv', corr_lag1, delimiter=',', 
           header=','.join(network_names), comments='')
np.savetxt('corr_lag2.csv', corr_lag2, delimiter=',',
           header=','.join(network_names), comments='')

print("✅ Matrices de correlación guardadas en corr_lag1.csv y corr_lag2.csv")

# ------------------- Heatmaps -------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(corr_lag1, annot=True, fmt='.3f', cmap='coolwarm', 
            xticklabels=network_names, yticklabels=network_names, 
            cbar_kws={'label': 'Correlation (Lag 1)'})
plt.title('Cross-Correlation (t -> t+1)')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.heatmap(corr_lag2, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=network_names, yticklabels=network_names,
            cbar_kws={'label': 'Correlation (Lag 2)'})
plt.title('Cross-Correlation (t -> t+2)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('cross_correlation_heatmaps.png', dpi=300)
plt.show()
print("✅ Heatmaps guardados en cross_correlation_heatmaps.png")