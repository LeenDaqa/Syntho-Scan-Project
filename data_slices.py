import nibabel as nib
import matplotlib.pyplot as plt
import os

# Define the main directory containing all subdirectories with NIfTI images
main_directory = r'/home/user/Gp sCT/PyTorch-CycleGAN-master/Task1/brain'

# Define the base save paths for CT and MR slices
base_ct_save_path = r'/home/user/Gp sCT/PyTorch-CycleGAN-master/ct_slices'
base_mr_save_path = r'/home/user/Gp sCT/PyTorch-CycleGAN-master/mri_slices'

# Ensure the base save paths exist
os.makedirs(base_ct_save_path, exist_ok=True)
os.makedirs(base_mr_save_path, exist_ok=True)

# Iterate through all subdirectories and files in the main directory
for root, dirs, files in os.walk(main_directory):
    for file in files:
        # Check if the file is a NIfTI image (by extension)
        if file.endswith('ct.nii.gz') or file.endswith('mr.nii.gz'):
            file_path = os.path.join(root, file)
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # Exclude the first and last 25 slices
            start_slice = 25
            end_slice = img_data.shape[2] - 25

            # Determine the modality and select the corresponding base save path
            modality = 'ct' if 'ct.nii.gz' in file else 'mri'
            base_save_path = base_ct_save_path if modality == 'ct' else base_mr_save_path

            # Patient ID for creating a unique name for each slice
            patient_id = os.path.basename(root)

            # Iterate through the selected slices
            for slice_index in range(start_slice, end_slice):
                slice_data = img_data[:, :, slice_index]

                # Name each slice using the patient ID, modality, and slice index
                unique_name = f"{patient_id}_{modality}_{slice_index}.jpg"
                output_file_path = os.path.join(base_save_path, unique_name)

                # Save the slice as a JPEG
                plt.imsave(output_file_path, slice_data, cmap='gray')
                print(f"Saved slice {slice_index} from {file_path} as '{output_file_path}'.")
