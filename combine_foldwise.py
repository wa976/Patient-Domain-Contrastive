# Reading the contents of the files
hospital_filename = './data/hospital_foldwise.txt'
iphone_filename = './data/iphone_foldwise_new.txt'

with open(hospital_filename, 'r') as file:
    hospital_data = file.readlines()

with open(iphone_filename, 'r') as file:
    iphone_data = file.readlines()

# Extracting the index (number) part from each line in iphone data
iphone_indices = set(line.split()[0] for line in iphone_data)
hospital_indices = set(line.split()[0] for line in hospital_data)

print(len(iphone_indices))
print(len(hospital_indices))

# Filter hospital data to only include lines where the index is not in iphone data
combined_data = [line for line in hospital_data if line.split()[0] not in iphone_indices]

combined_indices = set(line.split()[0] for line in combined_data)

# Combine the filtered hospital data with the iphone data
final_combined_data = iphone_data + combined_data

final_indices = set(line.split()[0] for line in final_combined_data)

print(len(combined_indices))
print(len(final_indices))


# Define the filename for the new combined file
new_filename = './data/total_foldwise_new.txt'

# Write the combined data to the new file
with open(new_filename, 'w') as file:
    file.writelines(final_combined_data)
