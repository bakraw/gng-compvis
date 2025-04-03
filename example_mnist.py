import gng
import visualization
import torch
import torchvision


# Transformations to apply.
# Gaussian blurring then downsampling to 14x14.
# Normalization is done by subtracting the mean and dividing by the standard deviation.
# 0.1307 and 0.3081 are the mean and standard deviations of the MNIST dataset.
# Every example I've seen uses those two values so I guess they work.
transform = torchvision.transforms.Compose([
    torchvision.transforms.GaussianBlur(3),
    torchvision.transforms.Resize((14, 14)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# Fetch the MNIST dataset.
training_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testing_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders.
# GNG being purely sequential, we have to iterate through images one by one.
# As such, a batch size over 1000 will result in a negligible speed increase.
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1000, shuffle=True)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1000, shuffle=True)

# Create the GNG model.
model = gng.Gng(14*14, 0.1, 0.03, 30, 40, 0.5, 0.995, 1,  100, "cpu")
model.train(training_dataloader, class_count=10)
model.test(testing_dataloader)

# Visualization.
visualization = visualization.Visualization(model)
visualization.pca("test_pca.png", third_dim=True)
visualization.mds("test_mds.png", third_dim=False)
visualization.tsne("test_tsne.png", third_dim=False)
visualization.umap("test_umap.png", third_dim=True)