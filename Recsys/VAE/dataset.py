import torch
from torchvision import datasets,transforms

def get_dataloader(args):

    transform = transforms.Compose([
                transforms.ToTensor()
                ])

    train = datasets.MNIST(root=args.dpath,train=True,transform=transform,download=True)
    test = datasets.MNIST(root=args.dpath,train=False,transform=transform,download=True)

    train_dataloader = torch.utils.data.DataLoader(
                train,
                batch_size=args.batch_size,
                shuffle=True,

    )

    test_dataloader = torch.utils.data.DataLoader(
                test,
                batch_size=args.batch_size,
                shuffle=False,
    )
    
    return train_dataloader,test_dataloader