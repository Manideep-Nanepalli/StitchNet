import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Input and output directories
src_base_dir = '/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/kits19/data'
dst_base_dir = '/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/data'

target_size = (256, 256)

for case in tqdm(sorted(os.listdir(src_base_dir))):
    case_path = os.path.join(src_base_dir, case)
    if not os.path.isdir(case_path):
        continue

    imaging_path = os.path.join(case_path, 'imaging.nii.gz')
    segmentation_path = os.path.join(case_path, 'segmentation.nii.gz')

    if not os.path.exists(imaging_path) or not os.path.exists(segmentation_path):
        print(f"Skipping {case} due to missing files.")
        continue

    # Load volumes
    img_nii = nib.load(imaging_path)
    seg_nii = nib.load(segmentation_path)

    img_data = img_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    # Check volume shapes
    if img_data.shape != seg_data.shape:
        print(f"Shape mismatch in {case}, skipping.")
        continue

    print(f"\n[{case}] Original shape - Imaging: {img_data.shape}, Segmentation: {seg_data.shape}")

    # Resize each 2D slice along the depth (axis 0)
    resized_img = []
    resized_seg = []

    for i in range(img_data.shape[0]):
        # Convert slices to PIL Images
        img_slice = Image.fromarray(img_data[i].astype(np.float32))
        seg_slice = Image.fromarray(seg_data[i].astype(np.uint8))

        # Resize
        img_resized = img_slice.resize(target_size, resample=Image.BILINEAR)
        seg_resized = seg_slice.resize(target_size, resample=Image.NEAREST)

        resized_img.append(np.array(img_resized))
        resized_seg.append(np.array(seg_resized))

    # Stack back into 3D volumes
    resized_img = np.stack(resized_img, axis=0)
    resized_seg = np.stack(resized_seg, axis=0)

    print(f"[{case}] Resized shape - Imaging: {resized_img.shape}, Segmentation: {resized_seg.shape}")

    # Save new NIfTI volumes in destination directory
    out_case_dir = os.path.join(dst_base_dir, case)
    os.makedirs(out_case_dir, exist_ok=True)

    img_nii_resized = nib.Nifti1Image(resized_img, affine=img_nii.affine)
    seg_nii_resized = nib.Nifti1Image(resized_seg, affine=seg_nii.affine)

    nib.save(img_nii_resized, os.path.join(out_case_dir, 'imaging.nii.gz'))
    nib.save(seg_nii_resized, os.path.join(out_case_dir, 'segmentation.nii.gz'))
