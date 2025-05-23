import torch
import torchvision
import gng

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.ColorJitter(saturation=1.5)
])

training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testing_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

training_dataloader = torch.utils.data.DataLoader(training_dataset, shuffle=True)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, shuffle=True)

model = gng.Gng(0.05, 0.006, 30, 30, 0.5, 0.995, 3, 32*32*3, 2500)
model.train(training_dataloader, 10)
model.test(testing_dataloader)