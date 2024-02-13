import os

def rename_jpg_to_jpeg(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the current file is a JPG file
        if filename.endswith(".jpg"):
            # Create the new file name by replacing .jpg with .jpeg
            new_filename = filename[:-4] + ".jpeg"
            
            # Create the full path for the original and new file names
            original_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(original_filepath, new_filepath)
            print(f"Renamed {original_filepath} to {new_filepath}")

# Specify the directory containing the JPG files
directory = './JPEG_Dataset/'

# Call the function with the specified directory
rename_jpg_to_jpeg(directory)
