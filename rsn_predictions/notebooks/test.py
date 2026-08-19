import pickle
import numpy as np

with open('state_codes_all.pkl', 'rb') as f:
    codes = pickle.load(f)

print(f"Total tokens: {len(codes)}")
print("Distribución de tokens:", np.bincount(codes))

# Convertir a binario y calcular tasas por red
binary = np.array([[(code >> b) & 1 for b in range(6, -1, -1)] for code in codes])
rates = binary.mean(axis=0)
print("Tasas de activación por red:", rates)