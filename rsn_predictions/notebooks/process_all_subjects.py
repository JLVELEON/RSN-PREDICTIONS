import os
import pickle
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import openneuro as on
import shutil
import logging
import datetime

# ============================================================
# 0. CONFIGURACIÓN DE PRUEBA (SOLO 3 SUJETOS)
# ============================================================
# 🔧 CAMBIA ESTA LISTA CON LOS 3 SUJETOS QUE QUIERAS PROBAR
# Asegúrate de que existan en el dataset. 
# Si no sabes cuáles, usa los que aparecían en los errores: 'sub-090', 'sub-100', 'sub-101'
TEST_SUBJECTS = ['sub-011', 'sub-012', 'sub-013']   # <--- PON AQUÍ LOS 3 QUE QUIERAS

# ============================================================
# 1. Configurar logging
# ============================================================
log_filename = f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# 2. Cargar las máscaras
# ============================================================
with open('mask_imgs_tutor.pkl', 'rb') as f:
    mask_imgs_tutor = pickle.load(f)

os.makedirs('./tmp/', exist_ok=True)

# ============================================================
# 3. Función para procesar un sujeto
# ============================================================
def process_subject(file_path):
    try:
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
    except Exception as e:
        print(f"    Error dentro de process_subject: {e}")
        raise

# ============================================================
# 4. Procesar los 3 sujetos de prueba (con reanudación)
# ============================================================
PROGRESS_FILE = 'progress_state_codes.pkl'
LAST_INDEX_FILE = 'last_index.txt'

all_state_codes = []
start_idx = 0
start_run = 1

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'rb') as f:
        all_state_codes = pickle.load(f)
    print(f"Progreso cargado: {len(all_state_codes)} tokens ya procesados.")

if os.path.exists(LAST_INDEX_FILE):
    with open(LAST_INDEX_FILE, 'r') as f:
        parts = f.read().strip().split()
        start_idx = int(parts[0])
        start_run = int(parts[1])
    print(f"Reanudando desde sujeto {TEST_SUBJECTS[start_idx]} (índice {start_idx}), run {start_run}")

print(f"Usando {len(TEST_SUBJECTS)} sujetos de prueba: {TEST_SUBJECTS}")

for idx in range(start_idx, len(TEST_SUBJECTS)):
    subject_id = TEST_SUBJECTS[idx]
    runs_to_process = range(start_run, 4) if idx == start_idx else range(1, 4)
    for run in runs_to_process:
        file_path = f"{subject_id}/func/{subject_id}_task-rest_run-{run:02d}_bold.nii.gz"
        try:
            print(f"Descargando y procesando {subject_id}, run-{run:02d}...")
            on.download(dataset='ds005747', target_dir='./tmp/', include=file_path)
            local_file = os.path.join('./tmp/', file_path)
            
            if not os.path.exists(local_file):
                print(f"  ⚠️ El archivo no se ha descargado: {local_file}")
                with open(LAST_INDEX_FILE, 'w') as f:
                    f.write(f"{idx} {run}")
                continue
            
            print(f"  ✅ Archivo descargado: {local_file}")
            
            try:
                codes = process_subject(local_file)
                print(f"  -> Códigos obtenidos: {len(codes)}")
            except Exception as e_proc:
                print(f"  ❌ Error en process_subject: {e_proc}")
                logging.error(f"Error en process_subject {subject_id} run-{run:02d}: {e_proc}")
                with open(PROGRESS_FILE, 'wb') as f:
                    pickle.dump(all_state_codes, f)
                with open(LAST_INDEX_FILE, 'w') as f:
                    f.write(f"{idx} {run}")
                continue
            
            all_state_codes.extend(codes.tolist())
            print(f"  -> {len(codes)} tokens añadidos (total: {len(all_state_codes)})")
            
            # Limpiar archivo y carpeta
            os.remove(local_file)
            shutil.rmtree(os.path.dirname(local_file), ignore_errors=True)
            
            # Guardar progreso después de cada run
            with open(PROGRESS_FILE, 'wb') as f:
                pickle.dump(all_state_codes, f)
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{idx} {run}")
                
        except FileNotFoundError:
            print(f"  -> {subject_id}, run-{run:02d} no encontrado (posiblemente no existe)")
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{idx} {run}")
        except Exception as e:
            print(f"  ❌ Error general: {e}")
            logging.error(f"Error en {subject_id}, run-{run:02d}: {e}")
            with open(PROGRESS_FILE, 'wb') as f:
                pickle.dump(all_state_codes, f)
            with open(LAST_INDEX_FILE, 'w') as f:
                f.write(f"{idx} {run}")
            # raise  # si quieres parar
    # Reiniciar run para el siguiente sujeto
    start_run = 1

# ============================================================
# 5. Guardar resultados
# ============================================================
all_state_codes = np.array(all_state_codes)
with open('state_codes_all.pkl', 'wb') as f:
    pickle.dump(all_state_codes, f)

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
if os.path.exists(LAST_INDEX_FILE):
    os.remove(LAST_INDEX_FILE)

print(f"\nProcesamiento completado.")
print(f"Total de tokens: {len(all_state_codes)}")
print(f"Guardado en: state_codes_all.pkl")