import os
import shutil

def copy_and_rename_files(source_folder, destination_folder, append_text):
    """
    Copies files from source folder to destination folder and renames them by appending text
    before the last character of the original file name.
    
    :param source_folder: Path of the source folder
    :param destination_folder: Path of the destination folder
    :param append_text: Text to append before the last character of the file name
    """
    import os
    import shutil

    # Ensure destination directory exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in os.listdir(source_folder):
        # Construct full file path
        source_file = os.path.join(source_folder, file_name)
        # Check if it's a file and not a directory
        if os.path.isfile(source_file):
            # Split file name and extension
            base_name, extension = os.path.splitext(file_name)
            # Insert append_text before the last character of the base name
            if base_name:
                new_file_name = f"{base_name[:-1]}{append_text}{base_name[-1]}{extension}"
            else:
                # If the base name is empty (unlikely, but could happen with a hidden file), use the append_text as the new name
                new_file_name = append_text + extension
            # Construct full destination file path
            destination_file = os.path.join(destination_folder, new_file_name)
            # Copy and rename file
            shutil.copy2(source_file, destination_file)

# Example usage
source_folder = './data/new_iphone/fold0/test/real'
destination_folder = 'data/total_dataset'
append_text = 'I'

# This function call will copy all files from the source folder to the destination folder
# and append '_new' to the end of each file name.
copy_and_rename_files(source_folder, destination_folder, append_text)