import os

def remove_non_image_files(directory):
    allowed_extensions = ['.jpg', '.png']
    
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            file_extension = os.path.splitext(file)[1].lower()  # Get file extension and convert to lowercase
            
            if file_extension not in allowed_extensions:
                try:
                    os.remove(filepath)
                    print(f"Removed: {filepath}")
                except Exception as e:
                    print(f"Could not remove {filepath}. Error: {e}")

# Replace 'path/to/image/folder' with the directory where you store your image files.
remove_non_image_files("C:/Users/Kaliber/Desktop/AI-Developer-Jensen/AI-kursen/Uppgifter/transferLearning/data/train/cat")
