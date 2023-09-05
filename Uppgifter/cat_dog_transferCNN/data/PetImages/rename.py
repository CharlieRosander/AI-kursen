import os

# Get the current working directory
folder_path = os.getcwd()

# Loop through every file in the directory
for filename in os.listdir(folder_path):
    # Check if the file is a .jpg file
    if filename.endswith(".jpg"):
        # Create the new filename by adding "cat_" prefix
        new_filename = "cat_" + filename
        
        # Full path for both old and new filename
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} to {new_file_path}")
