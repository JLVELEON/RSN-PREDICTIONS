from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib

# cargar fMRI
img = nib.load("data/sub--011/func/sub-011_task-rest_run-01_bold.nii.gz")

# cargar atlas Schaefer (100 regiones como ejemplo)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)

# crear extractor de señales
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True)

# extraer señales
time_series = masker.fit_transform(img)

print("Shape señales:", time_series.shape)