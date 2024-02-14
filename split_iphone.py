import os
import shutil

# Replace these with your actual directory paths
source_directory = './data/total/fold0/test/real'
destination_directory = './data/iphone/fold0/test/real'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Copy files that contain 'IN' in their names
for filename in os.listdir(source_directory):
    if 'I' in filename:
        source_file = os.path.join(source_directory, filename)
        destination_file = os.path.join(destination_directory, filename)
        shutil.copy(source_file, destination_file)
        print(f"Copied: {filename}")