import pickle as pkl


dataset_type = 'validation'


with open(f'./slices_per_patient_{dataset_type}.pkl', 'rb') as f:
    slices_per_patient = pkl.load(f)


print("Slices per patient in validation: ", slices_per_patient)

print("Total number of slices in validation: ", sum(slices_per_patient))

for slice in slices_per_patient:
    print(slice % 6)