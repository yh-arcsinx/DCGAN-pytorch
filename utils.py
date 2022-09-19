import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = '/home/gpu-3090/paimon/DCGAN-PyTorch-master/images'

def get_celeba(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        # transforms.LinearTransformation()
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))
        ])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)
    # b=dataset
    # for i in range(len(dataset)):
    #     dataset[i]=(dataset[i][0]/177.5-1,dataset[i][1])
    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader