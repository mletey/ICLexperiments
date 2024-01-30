import sys

sys.path.append('../../')
sys.path.append('../../../')

for i in range(30):
    if i != 40:
        file_path = f'../../../../resultssgd/longresults10d{i}.txt'
        with open(file_path, 'r') as file:
            file_contents = file.read()
        print(file_contents)

