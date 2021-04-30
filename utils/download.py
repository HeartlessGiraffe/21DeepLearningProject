import torchvision
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        transform=None, 
        target_transform=train_transform, 
        download=True)
torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=None, 
        target_transform=test_transform, 
        download=True)
