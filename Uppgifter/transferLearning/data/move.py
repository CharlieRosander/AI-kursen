import os
import shutil

# Get the current working directory (where the script is run from)
script_dir = os.getcwd()

# Navigate up two levels to get the base directory
base_dir = os.path.dirname(os.path.dirname(script_dir))

# Construct the full paths for the test, cat, and dog directories
test_dir = os.path.join(base_dir, 'data/test')
cat_dest = os.path.join(base_dir, 'data/test/cat')
dog_dest = os.path.join(base_dir, 'data/test/dog')

# Create the destination directories if they do not exist
if not os.path.exists(cat_dest):
    os.makedirs(cat_dest)
if not os.path.exists(dog_dest):
    os.makedirs(dog_dest)

# Iterate over each file in the test directory
for file in os.listdir(test_dir):
    # Skip directories
    if os.path.isdir(os.path.join(test_dir, file)):
        continue

    # Create the full path to the file
    full_file_path = os.path.join(test_dir, file)
    
    # Move the cat and dog images to their respective folders
    if 'cat' in file:
        shutil.move(full_file_path, os.path.join(cat_dest, file))
    elif 'dog' in file:
        shutil.move(full_file_path, os.path.join(dog_dest, file))
