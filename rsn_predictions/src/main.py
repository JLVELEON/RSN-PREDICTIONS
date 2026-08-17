import nibabel as nib

# cargar imagen
img = nib.load("data/sub--011/func/sub-011_task-rest_run-01_bold.nii.gz")

# convertir a array
data = img.dataobj 

# imprimir dimensiones
print("Dimensiones:", img.shape)

# test 
assert len(data.shape) == 4