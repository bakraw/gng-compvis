import gng
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
# GNG being purely sequential, we keep the default batch size (i.e. 1).
# We technically could set a higher value but it won't have any effect.
training_dataloader = torch.utils.data.DataLoader(training_dataset, shuffle=True)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, shuffle=True)

model = gng.Gng(0.2, 0.006, 50, 100, 0.5, 0.995, 2, 14*14, 200)
model.train(training_dataloader)
