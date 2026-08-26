import pickle
import numpy as np

with open('state_codes_all.pkl', 'rb') as f:
    codes = pickle.load(f)

print(f"Total tokens en progreso: {len(codes)}")
print(f"Tokens únicos: {len(np.unique(codes))}")
print(f"Distribución (primeros 10): {np.bincount(codes)[:10]}")

binary = np.array([[(code >> b) & 1 for b in range(6, -1, -1)] for code in codes])
rates = binary.mean(axis=0)
print(f"Tasas de activación por red: {rates}")