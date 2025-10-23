import pdb
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import ndimage
import os
from PIL import Image
import os
import nibabel as nib
import cv2
import SimpleITK as sitk

def _load_dicom_series(folder_path):
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

def _resize_volume(volume, is_mask=False):
    resized = []
    interp = cv2.INTER_NEAREST  # for both image and mask to keep label integrity
    for slice in volume:
        resized_slice = cv2.resize(slice, (256, 256), interpolation=interp)
        resized.append(resized_slice)
    return np.stack(resized, axis=0)

def test_all_case(net, dataset, semantic_class, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    total_array = np.zeros((len(image_list), 4))
    
    i = 0

    dice_lis = []
    hd95_lis = []
    # total_metric = []
    for image_path in tqdm(image_list):
        # h5f = h5py.File(image_path, 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        # image_dir = os.path.join(image_path, 'images')
        # mask_dir = os.path.join(image_path, 'masks')

        # slice_filenames = sorted(os.listdir(image_dir))

        # images = []
        # labels = []

        # for filename in slice_filenames:
        #     img_path = os.path.join(image_dir, filename)
        #     mask_path = os.path.join(mask_dir, filename)

        #     img = Image.open(img_path)
        #     mask = Image.open(mask_path)

        #     img_array = np.array(img, dtype=np.float32)
        #     mask_array = np.array(mask, dtype=np.uint8)

        #     images.append(img_array)
        #     labels.append(mask_array)

        # image_volume = np.stack(images, axis=0)
        # label_volume = np.stack(labels, axis=0)

        # ircad

        # ct_path = os.path.join(image_path, "PATIENT_DICOM")
        # mask_path = os.path.join(image_path, "MASKS_DICOM", "liver")

        # image = _load_dicom_series(ct_path)
        # label = _load_dicom_series(mask_path)

        # image_volume = _resize_volume(image)
        # label_volume = _resize_volume(label, is_mask=True)
        

        # atlas

        image_volume = nib.load(os.path.join(image_path, "imaging.nii.gz")).get_fdata()
        label_volume = nib.load(os.path.join(image_path, "segmentation.nii.gz")).get_fdata()


        '''till this line'''
        label_volume[label_volume > 0] = 1
        
        # Change to [H, W, D] format
        
        image = np.transpose(image_volume, (1, 2, 0))
        label = np.transpose(label_volume, (1, 2, 0))
        
        # import matplotlib.pyplot as plt

        # cols = 8
        # rows = (label.shape[2] + cols - 1) // cols
        # fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))

        # for i in range(rows * cols):
        #     ax = axes[i // cols, i % cols]
        #     if i < label.shape[2]:
        #         ax.imshow(label[:, :, i], cmap='gray')
        #         ax.set_title(f'Slice {i}')
        #     ax.axis('off')

        # plt.tight_layout()
        # plt.show()
        

        if dataset == "la" or dataset == 'duke' or dataset == 'atlas' or dataset == 'kits19':
            # id = image_path.split('/')[-2] + "_" + image_path.split('/')[-1]
            id = image_path.split('/')[-1][5:]   # for atlas

            image = (image - np.mean(image)) / np.std(image)
            
        elif dataset == "pancreas" or dataset == 'btcv' or dataset == 'mact':
            id = image_path.split('.')[-2].split('/')[-1]
            image = (image - np.mean(image)) / np.std(image)
        elif dataset == "lits" or dataset == "kits" or dataset == "promise" or dataset == "acdc" or dataset == "brats":
            id = image_path.split('.')[-2].split('/')[-1]
            image = image.swapaxes(0, 2)
            label = label.swapaxes(0, 2)
            image = (image - np.mean(image)) / np.std(image)
            if semantic_class == 'tumor':
                label[label != 2] = 0
                label[label == 2] = 1
            else:
                label[label > 0] = 1

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        # print(prediction.shape, label.shape)
        # raise Exception
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])  # (175, 132, 88), (175, 132, 88)

        print(image.shape, prediction.shape, label.shape)
        print(np.unique(image), np.unique(prediction), np.unique(label))

        label_255 = (label * 255).astype(np.uint8)
        prediction_255 = (prediction * 255).astype(np.uint8)

        # --- Save all three volumes into one .pt file ---
        save_dict = {
            "image": image,               # (256,256,depth)
            "label": label_255,           # (256,256,depth)
            "prediction": prediction_255  # (256,256,depth)
        }
        torch.save(save_dict, f"atlas_unet++_volume_{48+i}.pt")

        dice_lis.append(single_metric[0])
        hd95_lis.append(single_metric[2])
        
        import matplotlib.pyplot as plt

        def visualize_predictions_and_labels(prediction, label, num_slices=16):
            assert prediction.shape == label.shape, "Shape mismatch between prediction and label"
            H, W, D = prediction.shape
            # Choose 16 evenly spaced slice indices
            indices = np.linspace(0, D - 1, num_slices, dtype=int)

            # Set up plot
            fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 1.5, 3.5))
            for i, idx in enumerate(indices):
                axes[0, i].imshow(label[:, :, idx], cmap='gray')
                axes[0, i].set_title(f"Label\nSlice {idx}")
                axes[0, i].axis('off')

                axes[1, i].imshow(prediction[:, :, idx], cmap='gray')
                axes[1, i].set_title(f"Pred\nSlice {idx}")
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.show()

        # Example usage:
        # visualize_predictions_and_labels(prediction, label)

        # continue
        
        # print(id, single_metric)
        total_metric += np.asarray(single_metric)
        total_array[i] = single_metric
        i += 1

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(score_map[1].astype(np.float32), np.eye(4)), test_save_path + id + "_score.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")

    avg_metric = total_metric / len(image_list)
    std = np.std(total_array, axis=0, ddof=1)
    print('average metric is {}'.format(avg_metric))
    print('average std is {}'.format(std))
    # print(dice_lis, sum(dice_lis)/len(dice_lis))
    # print(hd95_lis, sum(hd95_lis)/len(hd95_lis))

    # import pandas as pd

    # df = pd.read_csv(r"/mnt/ab146a07-47de-444d-8709-50981a4043c2/research-contributions/volume_wise_dice_ircad.csv")
    # df['unet++'] = dice_lis
    # df.to_csv(r"/mnt/ab146a07-47de-444d-8709-50981a4043c2/research-contributions/volume_wise_dice_ircad.csv", index=False)

    # df = pd.read_csv(r"/mnt/ab146a07-47de-444d-8709-50981a4043c2/research-contributions/volume_wise_hd95_ircad.csv")
    # df['unet++'] = hd95_lis
    # df.to_csv(r"/mnt/ab146a07-47de-444d-8709-50981a4043c2/research-contributions/volume_wise_hd95_ircad.csv", index=False)

    # # raise Exception
    # median_dice = np.median(dice_lis)
    # q1_dice, q3_dice = np.percentile(dice_lis, [25, 75])

    # # HD95
    # median_hd95 = np.median(hd95_lis)
    # q1_hd95, q3_hd95 = np.percentile(hd95_lis, [25, 75])

    # print(f"Dice Median (IQR): {median_dice:.4f} ({q1_dice:.4f} - {q3_dice:.4f})")
    # print(f"HD95 Median (IQR): {median_hd95:.4f} ({q1_hd95:.4f} - {q3_hd95:.4f})")


    return avg_metric, std


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import random

    # # Assume `volume` is your input image of shape (256, 256, 83)
    # # For demonstration, here's a dummy volume
    # # volume = np.random.rand(256, 256, 83)

    # def visualize_random_slices(volume, num_slices=16):
    #     assert volume.ndim == 3, "Input volume should be 3D (H, W, D)"
    #     H, W, D = volume.shape
    #     selected_indices = sorted(random.sample(range(D), num_slices))

    #     fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    #     for ax, idx in zip(axes.flat, selected_indices):
    #         ax.imshow(volume[:, :, idx], cmap='gray')
    #         ax.set_title(f"Slice {idx}")
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()

    # Example usage:
    
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print(test_patch.shape)
                volume = test_patch[0, 0]  # shape: (256, 256, 16)

                # Create stitched image (4x4 grid of 16 slices)
                stitched = torch.zeros((1024, 1024))
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        stitched[i*256:(i+1)*256, j*256:(j+1)*256] = volume[:, :, idx]

                # Add back batch and channel dimensions
                test_patch = stitched.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 1024, 1024)

                # Convert to 3-channel image
                test_patch = test_patch.repeat(1, 3, 1, 1)
                test_patch = test_patch.cuda()
                # print(test_patch.shape)
                # import matplotlib.pyplot as plt
                # plt.imshow(test_patch[0].permute(1, 2, 0).cpu().numpy())
                # plt.axis('off')
                # plt.show()
                # raise Exception
                # y1, _, _, _ = net(test_patch)    # [1, 2, 112, 112, 80]
                batch = dict()
                batch['image'] = test_patch
                y1 = net.shared_step(batch)
                # print(y1[0]['semantic'].shape)
                y1 = y1[0]['semantic']
                y = F.softmax(y1, dim=1)
                # import matplotlib.pyplot as plt

                # y = y.detach()
                # mask = torch.argmax(y, dim=1)  # shape: (1, 1024, 1024)

                # plt.figure(figsize=(8, 8))
                # plt.imshow(mask[0].cpu().numpy(), cmap='gray')
                # plt.axis('off')
                # plt.show()
                B, C, H, W = y.shape  # (1, 2, 1024, 1024)
                assert H % 256 == 0 and W % 256 == 0, "Dimensions must be divisible by 256"
                
                grid_h = H // 256  # 4
                grid_w = W // 256  # 4
                
                # Step 1: Reshape to split into 4x4 grid of 256x256 tiles
                y = y.view(B, C, grid_h, 256, grid_w, 256)  # (1, 2, 4, 256, 4, 256)
                
                # Step 2: Bring tiles to front
                y = y.permute(0, 1, 3, 5, 2, 4).contiguous()  # (1, 2, 256, 256, 4, 4)
                
                # Step 3: Flatten grid back to 16 slices
                y = y.view(B, C, 256, 256, grid_h * grid_w)  # (1, 2, 256, 256, 16)
                
                # print(y.shape)
                # import matplotlib.pyplot as plt
                # # import torch

                # y = y.detach()
                # mask = torch.argmax(y, dim=1)  # shape: (1, 256, 256, 16)

                # fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                # mask_np = mask[0].cpu().numpy()  # shape: (256, 256, 16)

                # for i in range(16):
                #     ax = axes[i // 4, i % 4]
                #     ax.imshow(mask_np[:, :, i], cmap='gray')
                #     ax.axis('off')

                # plt.tight_layout()
                # plt.show()
                # raise Exception

                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)

    import matplotlib.pyplot as plt

    def plot_label_map_slices(label_map, slice_indices=None, cols=5):
        """
        Plots selected slices from a 3D label_map volume (shape: [H, W, D]).

        Args:
            label_map (np.ndarray): 3D array of shape [H, W, D].
            slice_indices (list of int, optional): List of slice indices along Z to plot. Defaults to 5 evenly spaced.
            cols (int): Number of columns in the plot grid.
        """
        assert label_map.ndim == 3, "label_map must be a 3D volume (H, W, D)"

        # Select default slices if not provided
        if slice_indices is None:
            total_slices = label_map.shape[2]
            slice_indices = list(range(0, total_slices, total_slices // 10))[:10]  # 10 slices

        rows = (len(slice_indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()

        for i, idx in enumerate(slice_indices):
            axes[i].imshow(label_map[:, :, idx], cmap='nipy_spectral')
            axes[i].set_title(f"Slice {idx}")
            axes[i].axis('off')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    # plot_label_map_slices(label_map)
    # raise Exception
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
