# import os

# def count_files_in_directory(directory):
#     return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# # Example usage
# directory_path = './data/total/fold0/training/real/'
# file_count = count_files_in_directory(directory_path)
# print(f"There are {file_count} files in the directory '{directory_path}'.")

# import torch
# torch.cuda.set_device(0) 
# print(torch.cuda.device_count())


import os

def count_files_with_label_N(directory):
    count = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # Check if the filename matches the format id-id2-label.wav and label is 'N'
            parts = filename.split('-')
            if len(parts) == 3 and parts[2] == 'HW.wav':
                count += 1
            # if len(parts) == 3 and parts[2] == 'N.wav':
            #     count += 1
    return count

# Example usage
directory_path = './data/total/fold0/training/real/'
file_count = count_files_with_label_N(directory_path)
print(f"There are {file_count} files in the directory '{directory_path}' with label 'N'.")

# import torch
# print(torch.version.cuda)