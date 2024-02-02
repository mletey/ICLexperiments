import sys

myname = sys.argv[1] # grab value of $mydir to add results
tauind = int(sys.argv[2]); # grab value of $SLURM_ARRAY_TASK_ID to index over taus 
avgind = int(sys.argv[3]); # grab value of $SLURM_ARRAY_TASK_ID to index over experiment repeats 

file_path = f'./{myname}/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'sup dude {tauind} + dont care {avgind}\n')
