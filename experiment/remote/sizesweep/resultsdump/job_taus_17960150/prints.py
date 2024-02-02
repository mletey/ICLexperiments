import os

# Specify the directory path
directory_path = './'

# Specify the prefix you're looking for
prefix = 'error'

# Get a list of all files in the directory
all_files = os.listdir(directory_path)

# Filter files that start with the specified prefix
filtered_files = [file for file in all_files if file.startswith(prefix)]
filtered_files = sorted(filtered_files)
# Loop over the filtered files
for file in filtered_files:
    print(file)
    # filepath = f'./{file}'
    # with open(filepath, 'r') as file:
    #     file_contents = file.read()
    # print(file_contents)
