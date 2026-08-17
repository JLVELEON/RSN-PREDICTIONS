import os
import pickle
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import openneuro as on
import shutil

# ============================================================
# 1. Cargar las máscaras
# ============================================================
with open('mask_imgs_tutor.pkl', 'rb') as f:
    mask_imgs_tutor = pickle.load(f)

os.makedirs('./tmp/', exist_ok=True)

# ============================================================
# 2. Función para procesar un sujeto
# ============================================================
def process_subject(file_path):
    img = nib.load(file_path)
    TR = 2.5
    time_series_red = {}
    for net in range(1, 8):
        masker = NiftiMasker(
            mask_img=mask_imgs_tutor[net],
            detrend=True,
            standardize='zscore_sample',
            low_pass=0.1,
            high_pass=0.01,
            t_r=TR,
            verbose=0
        )
        ts_voxels = masker.fit_transform(img)
        ts_mean = ts_voxels.mean(axis=1)
        time_series_red[net] = ts_mean
    time_series = np.column_stack([time_series_red[net] for net in range(1, 8)])
    time_series_norm = (time_series - time_series.mean(axis=0)) / time_series.std(axis=0)
    binary_states = (time_series_norm > 1.0).astype(int)
    weights = 2**np.arange(6, -1, -1)
    state_codes = binary_states @ weights
    return state_codes

# ============================================================
# 3. Descargar y procesar todos los sujetos (con reanudación)
# ============================================================
PROGRESS_FILE = 'progress_state_codes.pkl'
LAST_INDEX_FILE = 'last_index.txt'

# Cargar progreso anterior si existe
all_state_codes = []
start_subject = 1
start_run = 1

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'rb') as f:
        all_state_codes = pickle.load(f)
    print(f"Progreso cargado: {len(all_state_codes)} tokens ya procesados.")

if os.path.exists(LAST_INDEX_FILE):
    with open(LAST_INDEX_FILE, 'r') as f:
        parts = f.read().strip().split()
        start_subject = int(parts[0])
        start_run = int(parts[1])
    print(f"Reanudando desde sujeto {start_subject}, run {start_run}")

# Procesar sujetos
for i in range(start_subject, 91):
    subject_id = f"sub-{i:03d}"
    runs_to_process = range(start_run, 4) if i == start_subject else range(1, 4)
    for run in runs_to_process:
        file_path = f"{subject_id}/func/{subject_id}_task-rest_run-{run:02d}_bold.nii.gz"
        try:
            print(f"Descargando y procesando {subject_id}, run-{run:02d}...")
            on.download(dataset='ds005747', target_dir='./tmp/', include=file_path)
            local_file = os.path.join('./tmp/', file_path)
            codes = process_subject(local_file)
            all_state_codes.extend(codes.tolist())
            print(f"  -> {len(codes)} tokens añadidos (total: {len(all_state_codes)})")
            
            # Limpiar archivo y carpeta
            os.remove(local_file)
            shutil.rmtree(os.path.dirname(local_file), ignore_errors=True)
            
            # Guardar progreso después de cada run
            with open(PROGRESS_FILE, 'wb') as f:
                pickle.dump(all_state_codes, f)
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{i} {run}")
                
        except FileNotFoundError:
            print(f"  -> {subject_id}, run-{run:02d} no encontrado (posiblemente no existe)")
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{i} {run}")
        except Exception as e:
            print(f"  -> ERROR con {subject_id}, run-{run:02d}: {e}")
            with open(PROGRESS_FILE, 'wb') as f:
                pickle.dump(all_state_codes, f)
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{i} {run}")
            # Si quieres detener completamente en caso de error grave, descomenta:
            # raise
    # Reiniciar run para los siguientes sujetos
    start_run = 1

# ============================================================
# 4. Guardar todos los state_codes al final
# ============================================================
all_state_codes = np.array(all_state_codes)
with open('state_codes_all.pkl', 'wb') as f:
    pickle.dump(all_state_codes, f)

# Limpiar archivos de progreso (opcional)
if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
if os.path.exists(LAST_INDEX_FILE):
    os.remove(LAST_INDEX_FILE)

print(f"\nProcesamiento completado.")
print(f"Total de tokens: {len(all_state_codes)}")
print(f"Guardado en: state_codes_all.pkl")