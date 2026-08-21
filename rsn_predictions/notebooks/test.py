import pickle
import numpy as np

with open('state_codes_all.pkl', 'rb') as f:
    codes = pickle.load(f)

print(f"Total tokens: {len(codes)}")
print(f"Tokens únicos: {len(np.unique(codes))}")
print(f"Distribución de tokens (primeros 10): {np.bincount(codes)[:10]}")

# Convertir a binario para ver tasas de activación por red
binary = np.array([[(code >> b) & 1 for b in range(6, -1, -1)] for code in codes])
rates = binary.mean(axis=0)
print(f"Tasas de activación por red: {rates}")