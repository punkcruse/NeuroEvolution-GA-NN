from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_dataset(name='CIFAR10', batch_size=64, val_fraction=0.1):
    # 1) Define per‐dataset transforms
    if name.upper() == 'CIFAR10':
        # CIFAR-10 has 3 channels
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.CIFAR10
        input_dim  = 3 * 32 * 32
        output_dim = 10

    else:  # MNIST
        mean = (0.1307,)
        std  = (0.3081,)
        train_tf = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_cls = datasets.MNIST
        input_dim  = 1 * 28 * 28
        output_dim = 10

    # 2) Download full train+test splits
    full_train = dataset_cls(root='./data', train=True,  download=True, transform=train_tf)
    test_set   = dataset_cls(root='./data', train=False, download=True, transform=test_tf)

    # 3) Split off a validation set from the training data
    val_size   = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # 4) DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, input_dim, output_dim
