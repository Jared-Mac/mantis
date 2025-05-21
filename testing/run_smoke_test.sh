#!/bin/bash

# Navigate to the script's directory to ensure relative paths work correctly
# (Assuming this script is in 'testing/' and project root is parent)
# cd "$(dirname "$0")/.."

echo "---------------------------------------"
echo "STEP 1: Creating tiny CIFAR-100 dataset"
echo "---------------------------------------"
python testing/create_tiny_cifar100.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create tiny dataset. Exiting."
    exit 1
fi
echo "Tiny CIFAR-100 dataset created successfully."
echo ""

echo "---------------------------------------"
echo "STEP 2: Running end-to-end smoke test"
echo "---------------------------------------"
# Using --world_size 1 and --dist_url to simulate a single-GPU run, 
# which is common for local tests, even if not strictly DDP.
# This also makes it more robust if DDP settings are present in the config by default.
python main_classification_torchdistill.py \
    --config config/cifar100_film_chunked_tiny_test.yaml \
    --log_config \
    --disable_cudnn_benchmark \
    --world_size 1 \
    --dist_url 'env://' 
    # Adding --device cpu can also be an option for more universal testing if GPUs are an issue:
    # --device cpu

if [ $? -eq 0 ]; then
    echo ""
    echo "---------------------------------------"
    echo "SMOKE TEST SUCCEEDED"
    echo "---------------------------------------"
    echo "The script completed 1 epoch without crashing."
else
    echo ""
    echo "---------------------------------------"
    echo "SMOKE TEST FAILED"
    echo "---------------------------------------"
    echo "The script crashed or exited with an error."
    exit 1
fi

exit 0
