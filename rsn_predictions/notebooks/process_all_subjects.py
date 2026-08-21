import os
import sys
import pickle
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
import shutil
import logging
import datetime
import subprocess
import concurrent.futures
import warnings

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
# Pon True para procesar TODOS los sujetos, False para usar TEST_SUBJECTS
PROCESS_ALL = True   # Cambia a False si quieres una lista manual

# Lista manual (solo se usa si PROCESS_ALL = False)
TEST_SUBJECTS = ['sub-011', 'sub-012', 'sub-013']

# Número de descargas simultáneas (ajústalo)
MAX_WORKERS = 3

# ============================================================
# 1. Suprimir warnings de Nilearn y otras librerías
# ============================================================
warnings.filterwarnings('ignore')

# ============================================================
# 2. Configurar logging (solo errores graves)
# ============================================================
log_filename = f"processing_errors_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# 3. Cargar las máscaras
# ============================================================
with open('mask_imgs_tutor.pkl', 'rb') as f:
    mask_imgs_tutor = pickle.load(f)

os.makedirs('./tmp/', exist_ok=True)

# ============================================================
# 4. Función para procesar un sujeto (sin cambios, pero suprimimos warnings internos)
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
# 5. Encontrar la ruta de AWS CLI
# ============================================================
def get_aws_path():
    venv_aws_exe = os.path.join(sys.prefix, 'Scripts', 'aws.exe')
    if os.path.exists(venv_aws_exe):
        return venv_aws_exe
    aws_path = shutil.which('aws')
    if aws_path:
        return aws_path
    common_paths = [
        r'C:\Program Files\Amazon\AWSCLIV2\aws.exe',
        r'C:\Program Files (x86)\Amazon\AWSCLIV2\aws.exe',
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró AWS CLI.")

AWS_CMD = get_aws_path()
print(f"Usando AWS CLI en: {AWS_CMD}")

# ============================================================
# 6. Función para obtener todos los sujetos (usando AWS S3 ls)
# ============================================================
def get_all_subjects():
    cmd = [AWS_CMD, "s3", "ls", "s3://openneuro.org/ds005747/", "--no-sign-request"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    subjects = []
    for line in lines:
        # Salida típica: "                           PRE sub-090/"
        if "PRE sub-" in line:
            parts = line.split()
            # El último elemento es 'sub-090/'
            subject = parts[-1].strip('/')
            if subject.startswith('sub-'):
                subjects.append(subject)
    subjects.sort()
    return subjects

# ============================================================
# 7. Función de descarga con AWS CLI (muestra progreso)
# ============================================================
def download_with_aws(subject_id, run, target_dir='./tmp/'):
    file_path = f"{subject_id}/func/{subject_id}_task-rest_run-{run:02d}_bold.nii.gz"
    s3_path = f"s3://openneuro.org/ds005747/{file_path}"
    local_file = os.path.join(target_dir, file_path)
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    
    if os.path.exists(local_file) and os.path.getsize(local_file) > 100_000_000:
        print(f"  ⏩ {subject_id} run-{run:02d} ya existe ({os.path.getsize(local_file)/1e6:.1f} MB), se omite descarga.")
        return local_file
    elif os.path.exists(local_file):
        print(f"  ⚠️ {subject_id} run-{run:02d} existe pero es pequeño o corrupto, se descarga de nuevo.")
        os.remove(local_file)
    
    cmd = [AWS_CMD, "s3", "cp", "--no-sign-request", s3_path, local_file]
    print(f"  🚀 Descargando {subject_id} run-{run:02d} ...")
    subprocess.run(cmd, check=True)
    print(f"  ✅ Descargado {subject_id} run-{run:02d} ({os.path.getsize(local_file)/1e6:.1f} MB)")
    return local_file

# ============================================================
# 8. Función que procesa un sujeto/run completo (con manejo seguro de borrado)
# ============================================================
def process_one_run(subject_id, run):
    local_file = download_with_aws(subject_id, run)
    codes = process_subject(local_file)
    # Limpiar archivo después de procesar
    try:
        if os.path.exists(local_file):
            os.remove(local_file)
    except OSError:
        pass
    # NO ELIMINAR LA CARPETA COMPARTIDA
    return codes

# ============================================================
# 9. Procesamiento principal (con reanudación y paralelismo)
# ============================================================
# Si PROCESS_ALL = True, obtener todos los sujetos automáticamente
if PROCESS_ALL:
    print("Obteniendo lista de todos los sujetos...")
    TEST_SUBJECTS = get_all_subjects()
    if not TEST_SUBJECTS:
        raise RuntimeError("No se encontraron sujetos en el bucket.")
    print(f"Total de sujetos encontrados: {len(TEST_SUBJECTS)}")
    print("Primeros 10:", TEST_SUBJECTS[:10])

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

print(f"Usando {len(TEST_SUBJECTS)} sujetos.")
print(f"Descargas paralelas: {MAX_WORKERS} a la vez\n")

# Limpiar archivos corruptos en tmp (menores a 100MB)
for root, dirs, files in os.walk('./tmp/'):
    for file in files:
        if file.endswith('.nii.gz'):
            filepath = os.path.join(root, file)
            if os.path.getsize(filepath) < 100_000_000:
                print(f"  🗑️ Eliminando archivo corrupto: {filepath}")
                os.remove(filepath)

# Construir lista de tareas
tasks = []
for idx in range(start_idx, len(TEST_SUBJECTS)):
    subject_id = TEST_SUBJECTS[idx]
    runs = range(start_run, 4) if idx == start_idx else range(1, 4)
    for run in runs:
        tasks.append((subject_id, run, idx))

def run_task(subject_id, run, idx):
    try:
        codes = process_one_run(subject_id, run)
        return idx, codes, True, None
    except Exception as e:
        return idx, None, False, str(e)

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_task = {
        executor.submit(run_task, sub, run, idx): (sub, run, idx)
        for sub, run, idx in tasks
    }
    
    for future in concurrent.futures.as_completed(future_to_task):
        sub, run, idx = future_to_task[future]
        try:
            task_idx, codes, success, error = future.result()
            if success:
                all_state_codes.extend(codes.tolist())
                print(f"  ✅ {sub} run-{run:02d} procesado (total: {len(all_state_codes)})")
                with open(PROGRESS_FILE, 'wb') as f:
                    pickle.dump(all_state_codes, f)
                with open(LAST_INDEX_FILE, 'w') as f:
                    f.write(f"{idx} {run}")
            else:
                print(f"  ❌ {sub} run-{run:02d} falló: {error}")
                logging.error(f"Error en {sub} run-{run:02d}: {error}")
                with open(PROGRESS_FILE, 'wb') as f:
                    pickle.dump(all_state_codes, f)
                with open(LAST_INDEX_FILE, 'w') as f:
                    f.write(f"{idx} {run}")
        except Exception as e:
            print(f"  ❌ Error inesperado en {sub} run-{run:02d}: {e}")
            logging.error(f"Error inesperado en {sub} run-{run:02d}: {e}")

# ============================================================
# 10. Guardar resultados finales
# ============================================================
all_state_codes = np.array(all_state_codes)
with open('state_codes_all.pkl', 'wb') as f:
    pickle.dump(all_state_codes, f)

if os.path.exists(PROGRESS_FILE):
    os.remove(PROGRESS_FILE)
if os.path.exists(LAST_INDEX_FILE):
    os.remove(LAST_INDEX_FILE)

print(f"\n✅ Procesamiento completado.")
print(f"Total de tokens: {len(all_state_codes)}")
print(f"Guardado en: state_codes_all.pkl")