import pickle

configs = ['A_dropout', 'B_deeper', 'C_deeper_dropout']

for c in configs:
    try:
        with open(f'results_{c}.pkl', 'rb') as f:
            res = pickle.load(f)
        print(f"{c}: val_loss={res['val_losses'][-1]:.4f}, f1_changes={res['f1_changes']:.4f}, avg_f1={sum(res['f1_per_network'])/7:.4f}")
    except FileNotFoundError:
        print(f"{c}: archivo no encontrado")