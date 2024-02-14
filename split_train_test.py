
import os
import shutil

# Paths
data_folder = os.path.join("./data", 'total_dataset')
folds_file = os.path.join("./data", 'total_foldwise_new.txt')



test_fold = '4'
split = 'train'

dest_folder = os.path.join("./data/total/fold4")

# Create train and test directories
train_dir = os.path.join(dest_folder, 'training/real')
test_dir = os.path.join(dest_folder, 'test/real')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Processing filenames
filenames = os.listdir(data_folder)
filenames = set([f.strip().split('.')[0] for f in filenames if '.wav' in f])

# print(filenames)

train_flag = True
print_flag = True

if test_fold in ['0', '1', '2', '3', '4']:  # from RespireNet, 80-20% split
    patient_dict = {}
    all_patients = open(folds_file).read().splitlines()
    for line in all_patients:
        print(line)
        idx, fold = line.strip().split(' ')
        if train_flag and int(fold) != int(test_fold):
            patient_dict[idx] = fold
        elif train_flag and int(fold) == int(test_fold):
            patient_dict[idx] = fold
    
    if print_flag:
        print('*' * 20)
        print('Train and test 80-20% split with test_fold {}'.format(test_fold))
        print('Patience number in {} dataset: {}'.format(split, len(patient_dict)))


# print(patient_dict)


# Copying files to train and test folders
for f in filenames:
    # Extract patient index
    idx = f.split('-')[0]
    
    # print(idx)
    
    # Extract the third segment of the file name
    third_segment = f.split('-')[2]
    
    
    # # Clean and validate the index
    # idx = ''.join(filter(str.isdigit, idx))  # Keep only digits
    
    # print(idx)
    if idx not in patient_dict:
        # print(f"Skipping unknown index: {idx}")
        continue

    
    # for total
    if patient_dict[idx] == test_fold and third_segment[0] == 'I':
        dest_folder = test_dir
    elif patient_dict[idx] == test_fold and third_segment[0] == 'H':
        continue
    else:
        dest_folder = train_dir
    
    # Determine the destination folder
    if patient_dict[idx] == test_fold:
        dest_folder = test_dir
    else:
        dest_folder = train_dir

    # Copy the file
    src_path = os.path.join(data_folder, f + '.wav')
    dest_path = os.path.join(dest_folder, f + '.wav')
    if os.path.exists(src_path):  # Check if source file exists
        shutil.copy(src_path, dest_path)
    else:
        print(f"File not found: {src_path}")

# Print summary
print(f"Total files: {len(filenames)}")
print(f"Files moved to train folder: {len(os.listdir(train_dir))}")
print(f"Files moved to test folder: {len(os.listdir(test_dir))}")