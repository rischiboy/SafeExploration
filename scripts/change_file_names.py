import os

folder_path = "./stochastic_optimization/env_tests/data/horizon_test_same_env"

# List all files in the folder
files = os.listdir(folder_path)

new_names = ["horizon" + str(10 * i) + ".csv" for i in range(2, 8)]

# Rename files
for filename, new_name in zip(files, new_names):

    # Construct the full path for the old and new file names
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_path, new_path)
