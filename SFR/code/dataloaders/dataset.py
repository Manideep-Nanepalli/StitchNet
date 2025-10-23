import os
import pdb

import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk

def resampling(roiImg, new_size, lbl=False):
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(roiImg.GetSize(), roiImg.GetSpacing(), new_size)]
    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    return resampled_sitk



from PIL import Image
import torch

# class DukeLiverDataset(Dataset):
#     def __init__(self, base_dir, list_num='', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform

#         # Load the volume list from train{list_num}.list file
#         with open(os.path.join(self._base_dir, f'../train{list_num}.list'), 'r') as f:
#             self.volume_list = [line.strip() for line in f.readlines()]

#         if num is not None:
#             self.volume_list = self.volume_list[:num]

#         print(f"Total {len(self.volume_list)} volumes found.")

#     def __len__(self):
#         return len(self.volume_list)
                                                                                                    
#     def __getitem__(self, idx):
#         volume_name = self.volume_list[idx]  # e.g., "0001/11"
#         volume_path = os.path.join(self._base_dir, volume_name)

#         image_dir = os.path.join(volume_path, 'images')
#         mask_dir = os.path.join(volume_path, 'masks')

#         # Sort slice files to maintain consistent ordering
#         slice_filenames = sorted(os.listdir(image_dir))

#         images = []
#         labels = []

#         for filename in slice_filenames:
#             img_path = os.path.join(image_dir, filename)
#             mask_path = os.path.join(mask_dir, filename)

#             # Load images without converting — assuming grayscale PNGs
#             img = Image.open(img_path)
#             mask = Image.open(mask_path)

#             img_array = np.array(img, dtype=np.float32)
#             mask_array = np.array(mask, dtype=np.uint8)

#             images.append(img_array)
#             labels.append(mask_array)

#         # Stack into [D, H, W] volumes
#         image_volume = np.stack(images, axis=0)
#         label_volume = np.stack(labels, axis=0)
#         # print(image_volume.shape, label_volume.shape)
#         # raise Exception

#         image_volume = np.transpose(image_volume, (1, 2, 0))
#         label_volume = np.transpose(label_volume, (1, 2, 0))

#         # print("********", len(images), images[0].shape, image_volume.shape)

#         # Normalize the image volume
#         image_volume = (image_volume - np.mean(image_volume)) / np.std(image_volume)

#         sample = {
#             'image': image_volume.astype(np.float32),
#             'label': label_volume.astype(np.uint8)
            
#         }

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


class DukeLiverDataset(Dataset):
    def __init__(self, base_dir, list_num='', num=None, split='train', transform=None):
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        self._base_dir = base_dir
        self.transform = transform
        self.split = split

        # Load the full volume list
        list_path = os.path.join(self._base_dir, f'../train{list_num}.list')
        list_path = "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906/SFR_modeling/liver_data/train.list"
        with open(list_path, 'r') as f:
            volume_list = [line.strip() for line in f.readlines()]

        if num is not None:
            if split == 'train':
                self.volume_list = volume_list[:num]
            # elif split == 'val':
            #     self.volume_list = volume_list[num:num+62]
            # else:
            #     self.volume_list = volume_list[num + 62:]
            
        else:
            self.volume_list = volume_list[:260]

        print(f"{split.upper()} set: {len(self.volume_list)} volumes found.")
        
    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        volume_name = self.volume_list[idx]

        volume_path = os.path.join(self._base_dir, volume_name)

        image_dir = os.path.join(volume_path, 'images')
        mask_dir = os.path.join(volume_path, 'masks')

        slice_filenames = sorted(os.listdir(image_dir))

        images = []
        labels = []

        for filename in slice_filenames:
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            img = Image.open(img_path)
            mask = Image.open(mask_path)

            img_array = np.array(img, dtype=np.float32)
            mask_array = np.array(mask, dtype=np.uint8)

            images.append(img_array)
            labels.append(mask_array)

        image_volume = np.stack(images, axis=0)
        label_volume = np.stack(labels, axis=0)

        # Change to [H, W, D] format
        image_volume = np.transpose(image_volume, (1, 2, 0))
        label_volume = np.transpose(label_volume, (1, 2, 0))

        # Normalize
        image_volume = (image_volume - np.mean(image_volume)) / np.std(image_volume)

        sample = {
            'image': image_volume.astype(np.float32),
            'label': label_volume.astype(np.uint8)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# class Kits19(Dataset):
#     """ Kidney Dataset for binary segmentation (0 vs 1/2), with blank slices removed """
#     def __init__(self, base_dir=None, num=None, split='', transform=None):
#         self._base_dir = base_dir
#         self.transform = transform

#         # Read volume paths from the list file
#         list_path = os.path.join(self._base_dir, f'../train_kits.list')
#         with open(list_path, 'r') as f:
#             self.volume_list = [line.strip() for line in f.readlines()]

#         # if num is not None:
#         #     if split == 'train':
#         #         self.volume_list = self.volume_list[:num]   # e.g., 126 for training
#         #     elif split == 'val':
#         #         self.volume_list = self.volume_list[num:num + 42]
#         #     else:
#         #         self.volume_list = self.volume_list[num + 42:]
#         if num is not None:
#             self.volume_list = self.volume_list[:num]
        
#         print("total {} samples".format(len(self.volume_list)))

#     def _get_file_path(self, folder_path, filename_prefix):
#         """ Helper to find .nii or .nii.gz """
#         for ext in ['.nii.gz', '.nii']:
#             file_path = os.path.join(folder_path, filename_prefix + ext)
#             if os.path.exists(file_path):
#                 return file_path
#         raise FileNotFoundError(f"{filename_prefix}.nii or .nii.gz not found in {folder_path}")

#     def __len__(self):
#         return len(self.volume_list)

#     def __getitem__(self, idx):
#         volume_path = self.volume_list[idx]
#         image_path = self._get_file_path(volume_path, 'imaging')
#         label_path = self._get_file_path(volume_path, 'segmentation')

#         # Load NIfTI volumes
#         image = nib.load(image_path).get_fdata()
#         label = nib.load(label_path).get_fdata()

#         # Binarize label: convert 2 → 1
#         label[label > 0] = 1
        
        
#         # Remove blank slices: keep only slices where label has any non-zero value
#         # non_blank_slices = [i for i in range(label.shape[0]) if np.any(label[i, :, :])]
#         # image = image[non_blank_slices, :, :]
#         # label = label[non_blank_slices, :, :]

#         # Swap axes to match expected shape: (D, H, W) → (H, W, D)  
#         image = np.swapaxes(image, 0, 2)
#         label = np.swapaxes(label, 0, 2)
        
#         # Normalize image
#         image = (image - np.mean(image)) / np.std(image)

#         sample = {'image': image, 'label': label.astype(np.uint8)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class Kits19(Dataset):
    """ Kidney Dataset for binary segmentation (0 vs 1/2), with blank slices reduced by cropping """
    def __init__(self, base_dir=None, num=None, split='', transform=None):
        self._base_dir = base_dir
        self.transform = transform

        # Read volume paths from the list file
        list_path = os.path.join(self._base_dir, f'../train_kits.list')
        with open(list_path, 'r') as f:
            self.volume_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.volume_list = self.volume_list[:num]
        
        print("total {} samples".format(len(self.volume_list)))

    def _get_file_path(self, folder_path, filename_prefix):
        """ Helper to find .nii or .nii.gz """
        for ext in ['.nii.gz', '.nii']:
            file_path = os.path.join(folder_path, filename_prefix + ext)
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"{filename_prefix}.nii or .nii.gz not found in {folder_path}")

    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        volume_path = self.volume_list[idx]
        image_path = self._get_file_path(volume_path, 'imaging')
        label_path = self._get_file_path(volume_path, 'segmentation')

        # Load NIfTI volumes
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Binarize label: convert 2 → 1
        label[label > 0] = 1

        # Swap axes to match expected shape: (D, H, W) → (H, W, D)  
        image = np.swapaxes(image, 0, 2)
        label = np.swapaxes(label, 0, 2)

        # === CROP VOLUME TO REMOVE EXCESSIVE BLANK SLICES ===
        non_blank_indices = np.where(np.any(label > 0, axis=(0, 1)))[0]
        if len(non_blank_indices) > 0:
            start = non_blank_indices[0]
            end = non_blank_indices[-1]
            pad = 20
            start = max(0, start - pad)
            end = min(label.shape[2], end + pad + 1)
            image = image[:, :, start:end]
            label = label[:, :, start:end]
        # ====================================================

        # Normalize image
        image = (image - np.mean(image)) / np.std(image)

        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample



import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

import cv2
import os
import nibabel as nib
import numpy as np
import pydicom
from glob import glob
from torch.utils.data import Dataset


class AtlasIRCAD(Dataset):
    def __init__(self, base_dir=None, num=None, split='', transform=None):
        self._base_dir = base_dir
        self.transform = transform

        list_path = os.path.join(self._base_dir, 'train_ircad.list')
        with open(list_path, 'r') as f:
            self.volume_list = [line.strip() for line in f.readlines()]
        
        if num is not None:
            self.volume_list = self.volume_list[:num]

        print("Total {} samples loaded from train_ircad.list".format(len(self.volume_list)))

    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        folder_path = self.volume_list[idx]

        if 'ircad' in folder_path.lower():
            # IRCAD example:
            # CT path = ircad/3Dircadb1.1/PATIENT_DICOM/
            # Mask path = ircad/3Dircadb1.1/MASKS_DICOM/liver/
            ct_path = os.path.join(folder_path, "PATIENT_DICOM")
            mask_path = os.path.join(folder_path, "MASKS_DICOM", "liver")

            image = self._load_dicom_series(ct_path)
            label = self._load_dicom_series(mask_path)

            image = self._resize_volume(image)
            label = self._resize_volume(label, is_mask=True)
        else:
            raise Exception
            # Non-IRCAD NIfTI volume
            image_path = self._get_file_path(folder_path, 'imaging')
            label_path = self._get_file_path(folder_path, 'segmentation')
            image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()

        # Binarize mask and normalize image
        label[label > 0] = 1
        image = np.swapaxes(image, 0, 2)  # shape: (slices, H, W)
        label = np.swapaxes(label, 0, 2)
        image = (image - np.mean(image)) / np.std(image)

        sample = {'image': image.astype(np.float32), 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_file_path(self, folder_path, filename_prefix):
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(folder_path, filename_prefix + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{filename_prefix}.nii[.gz] not found in {folder_path}")

    def _load_dicom_series(self, folder_path):
        """Load a DICOM series using SimpleITK from a folder like:
           ircad/3Dircadb1.1/PATIENT_DICOM or ircad/3Dircadb1.1/MASKS_DICOM/liver
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        if len(dicom_names) == 0:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image)  # shape: [slices, H, W]

    def _resize_volume(self, volume, is_mask=False):
        resized = []
        interp = cv2.INTER_NEAREST  # for both image and mask to keep label integrity
        for slice in volume:
            resized_slice = cv2.resize(slice, (256, 256), interpolation=interp)
            resized.append(resized_slice)
        return np.stack(resized, axis=0)


class Atlas23(Dataset):
    """ Atlas Dataset for binary segmentation (0 vs 1), no blank slice removal """
    def __init__(self, base_dir=None, num=None, split='', transform=None):
        self._base_dir = base_dir
        self.transform = transform

        # Path to list file
        list_path = os.path.join(self._base_dir, 'train_atlas.list')
        with open(list_path, 'r') as f:
            self.volume_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.volume_list = self.volume_list[:num]

        print("Total {} samples loaded from train_atlas.list".format(len(self.volume_list)))

    def _get_file_path(self, folder_path, filename_prefix):
        """ Helper to find .nii or .nii.gz """
        for ext in ['.nii.gz', '.nii']:
            file_path = os.path.join(folder_path, filename_prefix + ext)
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"{filename_prefix}.nii or .nii.gz not found in {folder_path}")

    def __len__(self):
        return len(self.volume_list)

    def __getitem__(self, idx):
        volume_path = self.volume_list[idx]
        image_path = self._get_file_path(volume_path, 'imaging')
        label_path = self._get_file_path(volume_path, 'segmentation')

        # Load NIfTI volumes
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Ensure binary labels
        label[label > 0] = 1

        image = np.swapaxes(image, 0, 2)
        label = np.swapaxes(label, 0, 2)

        # Normalize image
        image = (image - np.mean(image)) / np.std(image)

        

        sample = {'image': image.astype(np.float32), 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


# class Kits19(Dataset):
#     def __init__(self, base_dir, list_num='', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform

#         # Load the volume list from train{list_num}.list file
#         with open(os.path.join(self._base_dir, f'../train{list_num}.list'), 'r') as f:
#             self.volume_list = [line.strip() for line in f.readlines()]

#         if num is not None:
#             self.volume_list = self.volume_list[:num]

#         print(f"Total {len(self.volume_list)} volumes found.")

#     def __len__(self):
#         return len(self.volume_list)
                                                                                                    
#     def __getitem__(self, idx):
#         volume_name = self.volume_list[idx]  # e.g., "0001/11"
#         volume_path = os.path.join(self._base_dir, volume_name
#         image_dir = os.path.join(volume_path, 'images')
#         mask_dir = os.path.join(volume_path, 'masks')

#         # Sort slice files to maintain consistent ordering
#         slice_filenames = sorted(os.listdir(image_dir))

#         images = []
#         labels = []
#         for filename in slice_filenames:
#             img_path = os.path.join(image_dir, filename)
#             mask_path = os.path.join(mask_dir, filename)

#             # Load images without converting — assuming grayscale PNGs
#             img = Image.open(img_path)
#             mask = Image.open(mask_path)

#             img_array = np.array(img, dtype=np.float32)
#             mask_array = np.array(mask, dtype=np.uint8)

#             images.append(img_array)
#             labels.append(mask_array)

#         # Stack into [D, H, W] volumes
#         image_volume = np.stack(images, axis=0)
#         label_volume = np.stack(labels, axis=0) 
#         # print(image_volume.shape, label_volume.shape)
#         # raise Exception

#         image_volume = np.transpose(image_volume, (1, 2, 0))
#         label_volume = np.transpose(label_volume, (1, 2, 0))

#         # print("********", len(images), images[0].shape, image_volume.shape)

#         # Normalize the image volume
#         image_volume = (image_volume - np.mean(image_volume)) / np.std(image_volume)

#         sample = {
#             'image': image_volume.astype(np.float32),
#             'label': label_volume.astype(np.uint8)
#         }

#         if self.transform:
#             sample = self.transform(sample)

#         return sample




class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, list_num='', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        with open(self._base_dir+'/../train' + list_num + '.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        image = (image - np.mean(image)) / np.std(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class LAHeart_unlab(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, list_num='', label_num=None):
        self._base_dir = base_dir
        self.sample_list = []

        with open(self._base_dir+'/../train' + list_num + '.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if label_num is not None:
            self.image_list = self.image_list[label_num:]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image, label = h5f['image'][:], h5f['label_full'][:]
        image = (image - np.mean(image)) / np.std(image)
        sample = {'name': image_name, 'image': image, 'label': label}
        # sample = {'image': image, 'label': label}
        return sample


class BTCV(Dataset):
    """ BTCV Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform

        # with open(self._base_dir+'/../train.list', 'r') as f:
        with open(self._base_dir+'/../train_magic.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = self._base_dir + '/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]         # (314, 314, 235)
        image = (image - np.mean(image)) / np.std(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MACT(Dataset):
    """ MACT Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform

        with open(self._base_dir+'/../train.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = self._base_dir + '/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]         # (314, 314, 235)
        image = (image - np.mean(image)) / np.std(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class BraTS19(Dataset):
    """ BraTS2019 Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        with open(self._base_dir + '/../train_follow.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}.h5".format(image_name), 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        image = image.swapaxes(0, 2)
        label = label.swapaxes(0, 2)
        image = (image - np.mean(image)) / np.std(image)
        label[label > 0] = 1
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class BraTS19_unlab(Dataset):
    """ BraTS2019 Dataset """
    def __init__(self, base_dir=None, label_num=None):
        self._base_dir = base_dir
        self.sample_list = []

        with open(self._base_dir + '/../train_follow.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if label_num is not None:
            self.image_list = self.image_list[label_num:]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}.h5".format(image_name), 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        image = image.swapaxes(0, 2)
        label = label.swapaxes(0, 2)
        image = (image - np.mean(image)) / np.std(image)
        label[label > 0] = 1
        sample = {'image': image, 'label': label}
        return sample


class Resample(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        new_size = self.output_size

        image_itk = resampling(sitk.GetImageFromArray(image), new_size, lbl=False)
        label_itk = resampling(sitk.GetImageFromArray(label), new_size, lbl=True)
        image = sitk.GetArrayFromImage(image_itk)
        label = sitk.GetArrayFromImage(label_itk)

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

# import numpy as np

# class RandomCrop(object):
#     """
#     Random crop 3D volumes with smart depth sampling.
    
#     If depth > desired depth: apply dynamic striding with random start.
#     If depth < desired depth: pad to required size.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (tuple, list)) and len(output_size) == 3, \
#             "output_size should be a tuple/list of (W, H, D)"
#         self.output_size = output_size  # (w, h, d)

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         w_t, h_t, d_t = self.output_size

#         # Pad if necessary
#         pad_w = max((w_t - image.shape[0]) // 2 + 3, 0)
#         pad_h = max((h_t - image.shape[1]) // 2 + 3, 0)
#         pad_d = max((d_t - image.shape[2]) // 2 + 3, 0)

#         if pad_w > 0 or pad_h > 0 or pad_d > 0:
#             image = np.pad(image, [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
#                            mode='constant', constant_values=0)
#             label = np.pad(label, [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
#                            mode='constant', constant_values=0)

#         w, h, d = image.shape

#         # Standard random crop in width and height
#         w1 = np.random.randint(0, w - w_t + 1)
#         h1 = np.random.randint(0, h - h_t + 1)
#         image = image[w1:w1 + w_t, h1:h1 + h_t, :]
#         label = label[w1:w1 + w_t, h1:h1 + h_t, :]

#         # Handle depth (slice) dimension
#         if d <= d_t:
#             # Not enough slices, pad in depth (already handled)
#             d1 = (image.shape[2] - d_t) // 2
#             image = np.pad(image, [(0, 0), (0, 0), (max(-d1, 0), max(d_t - image.shape[2] - max(-d1, 0), 0))],
#                            mode='constant', constant_values=0)
#             label = np.pad(label, [(0, 0), (0, 0), (max(-d1, 0), max(d_t - label.shape[2] - max(-d1, 0), 0))],
#                            mode='constant', constant_values=0)
#         else:
#             # Dynamic stride and strided sampling
#             stride = max(1, round(d / d_t))
#             start_idx = np.random.randint(0, stride)            
#             indices = [start_idx + i * stride for i in range(d_t)]
#             # Clip to avoid index overflow
#             indices = [min(idx, d - 1) for idx in indices]

#             image = image[:, :, indices]
#             label = label[:, :, indices]

#         return {'image': image, 'label': label}



class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
