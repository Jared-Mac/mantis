import os
import tarfile
import shutil
from pathlib import Path
import glob

def organize_imagenet(train_tar_path, val_tar_path, val_gt_txt_path, idx_to_synset_map_path, output_base_dir):
    """
    Organizes downloaded ImageNet ILSVRC2012 .tar files into the ImageFolder structure.

    Args:
        train_tar_path (str): Path to ILSVRC2012_img_train.tar
        val_tar_path (str): Path to ILSVRC2012_img_val.tar
        val_gt_txt_path (str): Path to ILSVRC2012_validation_ground_truth.txt
        idx_to_synset_map_path (str): Path to a text file mapping ILSVRC2012_IDs (1-1000) to WNIDs (nXXXXXXXX).
                                      Format per line: "<ILSVRC2012_ID> <WNID>"
        output_base_dir (str): The base directory where 'ilsvrc2012/train' and 'ilsvrc2012/val' will be created.
                               (e.g., '~/resources/datasets')
    """
    train_tar_path = Path(train_tar_path).expanduser()
    val_tar_path = Path(val_tar_path).expanduser()
    val_gt_txt_path = Path(val_gt_txt_path).expanduser()
    idx_to_synset_map_path = Path(idx_to_synset_map_path).expanduser()
    output_base_dir = Path(output_base_dir).expanduser()

    target_train_dir = output_base_dir / "ilsvrc2012" / "train"
    target_val_dir = output_base_dir / "ilsvrc2012" / "val"
    temp_train_extract_dir = output_base_dir / "ilsvrc2012" / "temp_train_extract"
    temp_val_extract_dir = output_base_dir / "ilsvrc2012" / "temp_val_extract"

    # --- Safety checks for input files ---
    if not train_tar_path.is_file():
        print(f"ERROR: Training TAR file not found at {train_tar_path}")
        return
    if not val_tar_path.is_file():
        print(f"ERROR: Validation TAR file not found at {val_tar_path}")
        return
    if not val_gt_txt_path.is_file():
        print(f"ERROR: Validation ground truth file not found at {val_gt_txt_path}")
        return
    if not idx_to_synset_map_path.is_file():
        print(f"ERROR: Index-to-synset mapping file not found at {idx_to_synset_map_path}")
        return

    print(f"Target training directory: {target_train_dir}")
    print(f"Target validation directory: {target_val_dir}")
    target_train_dir.mkdir(parents=True, exist_ok=True)
    target_val_dir.mkdir(parents=True, exist_ok=True)
    temp_train_extract_dir.mkdir(parents=True, exist_ok=True)
    temp_val_extract_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Process Training Data ---
    print(f"\nProcessing training data from {train_tar_path}...")
    try:
        with tarfile.open(train_tar_path, "r:") as main_tar:
            # Training images are in sub-tarfiles, one for each class (synset)
            synset_tars = [member for member in main_tar.getmembers() if member.name.endswith(".tar")]
            for i, synset_tar_info in enumerate(synset_tars):
                synset_id = synset_tar_info.name.split('.')[0]
                print(f"  Extracting synset {i+1}/{len(synset_tars)}: {synset_id}...")
                
                # Extract the inner synset.tar to a temporary location
                main_tar.extract(synset_tar_info, path=temp_train_extract_dir)
                inner_tar_path = temp_train_extract_dir / synset_tar_info.name
                
                # Create the target directory for this synset
                synset_output_dir = target_train_dir / synset_id
                synset_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract images from the inner synset.tar into its final directory
                with tarfile.open(inner_tar_path, "r:") as inner_tar:
                    inner_tar.extractall(path=synset_output_dir)
                
                # Remove the temporary inner synset.tar
                inner_tar_path.unlink()
        print("Training data processed successfully.")
    except Exception as e:
        print(f"An error occurred during training data processing: {e}")
    finally:
        if temp_train_extract_dir.exists():
            print(f"Cleaning up temporary training extraction directory: {temp_train_extract_dir}")
            shutil.rmtree(temp_train_extract_dir)

    # --- 2. Process Validation Data ---
    print(f"\nProcessing validation data from {val_tar_path}...")
    try:
        # Extract all validation images to a temporary flat directory
        print(f"  Extracting all validation images to {temp_val_extract_dir}...")
        with tarfile.open(val_tar_path, "r:") as val_tar:
            val_tar.extractall(path=temp_val_extract_dir)
        
        # Load ILSVRC2012_ID to WNID (synset) mapping
        print(f"  Loading ILSVRC2012_ID to WNID mapping from {idx_to_synset_map_path}...")
        id_to_synset = {}
        with open(idx_to_synset_map_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    id_to_synset[int(parts[0])] = parts[1]
        
        # Load validation ground truth
        print(f"  Loading validation ground truth from {val_gt_txt_path}...")
        with open(val_gt_txt_path, 'r') as f:
            val_ground_truth_ids = [int(line.strip()) for line in f.readlines()]

        # Get list of extracted validation images (should be sorted by name for correct mapping)
        extracted_val_images = sorted(list(temp_val_extract_dir.glob('ILSVRC2012_val_*.JPEG')))
        
        if len(extracted_val_images) != len(val_ground_truth_ids):
            print(f"ERROR: Mismatch in number of validation images ({len(extracted_val_images)}) "
                  f"and ground truth labels ({len(val_ground_truth_ids)}). Please check your files.")
            return

        print("  Organizing validation images into class subfolders...")
        for i, img_path in enumerate(extracted_val_images):
            ilsvrc_id = val_ground_truth_ids[i] # Ground truth is 1-indexed for image number, but list is 0-indexed
            synset_id = id_to_synset.get(ilsvrc_id)
            
            if synset_id:
                synset_dir = target_val_dir / synset_id
                synset_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(synset_dir / img_path.name))
            else:
                print(f"Warning: No synset ID found for ILSVRC2012_ID {ilsvrc_id} (image {img_path.name}). Skipping.")
            
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i+1}/{len(extracted_val_images)} validation images...")

        print("Validation data processed successfully.")
    except Exception as e:
        print(f"An error occurred during validation data processing: {e}")
    finally:
        if temp_val_extract_dir.exists():
            print(f"Cleaning up temporary validation extraction directory: {temp_val_extract_dir}")
            shutil.rmtree(temp_val_extract_dir)

    print("\nImageNet processing finished.")
    print(f"Data should be available in: {output_base_dir / 'ilsvrc2012'}")

if __name__ == '__main__':
    print("ImageNet ILSVRC2012 Dataset Organization Script")
    print("-" * 50)
    print("IMPORTANT: You must download the ILSVRC2012 .tar files manually after registering at image-net.org.")
    print("This script will help organize them into the ImageFolder format.")
    print("-" * 50)

    # --- User Inputs ---
    default_download_dir = Path.home() / "Downloads" # Common download location
    
    inp_train_tar = input(f"Enter full path to ILSVRC2012_img_train.tar (e.g., {default_download_dir / 'ILSVRC2012_img_train.tar'}): ").strip()
    inp_val_tar = input(f"Enter full path to ILSVRC2012_img_val.tar (e.g., {default_download_dir / 'ILSVRC2012_img_val.tar'}): ").strip()
    
    print("\nFor validation set organization, you need:")
    print("1. ILSVRC2012_validation_ground_truth.txt (from ILSVRC2012_devkit_t12.tar.gz)")
    print("2. A mapping file from ILSVRC2012_ID (1-1000) to WNID (e.g., n01234567).")
    print("   This mapping file ('idx_to_synset.txt') should have lines like: '1 n02129165'")
    inp_val_gt_txt = input(f"Enter full path to ILSVRC2012_validation_ground_truth.txt (e.g., {default_download_dir / 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'}): ").strip()
    inp_idx_to_synset_map = input(f"Enter full path to your idx_to_synset.txt mapping file: ").strip()
    
    default_output_dir = Path.home() / "resources" / "datasets"
    inp_output_base = input(f"Enter base output directory (default: {default_output_dir}): ").strip() or str(default_output_dir)

    organize_imagenet(inp_train_tar, inp_val_tar, inp_val_gt_txt, inp_idx_to_synset_map, inp_output_base)