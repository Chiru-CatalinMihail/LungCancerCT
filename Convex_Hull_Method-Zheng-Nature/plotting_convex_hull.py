import matplotlib
# matplotlib.use('Agg')


import matplotlib.pyplot as plt
import torch
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )

a = torch.load('pacienti_samples_squeezed/initial_image/patient_0.pt')
b = torch.load('pacienti_samples_squeezed/prediction_compact/patient_0.pt')

print(a.shape, b.shape)

# a = a.detach().numpy().astype(float)
# b = b.detach().numpy().astype(float)

# print(a.shape, type(a), a.dtype)

for i in range(2):
    a = torch.load(f'pacienti_samples_squeezed/initial_image/patient_{i}.pt')
    b = torch.load(f'pacienti_samples_squeezed/prediction_compact/patient_{i}.pt')

    for j in range(4):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(a[0, ..., j], cmap='gray')
        axes[1].imshow(b[0, ..., j])
        plt.savefig(f'./pacienti_samples_squeezed/gradient_matrix/patient_{i}_gradient_matrix_{j}.png')