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
    #torchvision.transforms.Resize((14, 14)),
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
model = gng.Gng(0.1, 0.03, 30, 40, 0.5, 0.995, 3, 28*28, 1700)
model.train(training_dataloader, 10)
model.test(testing_dataloader)
