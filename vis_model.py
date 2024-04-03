import argparse
from tqdm import tqdm
from functools import partial
import numpy as np
import torch 
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from plotting import make_gif

def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str, help='path of torch classifier to load')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--n_frames', default=100, type=int, help='number of frames in gif')
    parser.add_argument('--grid_size', default=1e-6, type=float, help='total size of grid range in image-space')
    parser.add_argument('--grid_samples', default=128, type=int, help='numer of samples to make grid height / width (output image size)')
    parser.add_argument('--save_path', default='output.gif', type=str, help='path to save gif')

    return parser.parse_args()

def load_model(model_path):
    if model_path is None:
        print('no model path provided, using default swin-t model')

        # modify last layer to output 10 classes
        # (for visualization clarity...)
        model = torchvision.models.swin_t()
        model.head = torch.nn.Linear(model.head.in_features, 10)
    else:
        print(f'loading model from {model_path}')
        model = torch.load(model_path)

    return model

def fetch_loader(batch_size, num_workers, split):
    print('fetching cifar-10 dataset')

    train = split == 'train'
    partial_dataset = partial(
        datasets.CIFAR10, root='./data',
        train=train, transform=transforms.ToTensor(),
    )

    try: dataset = partial_dataset(download=False)
    except: dataset = partial_dataset(download=True)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )

    return loader

# train swin on cifar-10 for a bit...
def train_swin(model, loader, n_epochs=10):
    print('training swin-t model on cifar-10')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        loss_track = []

        for x, y in tqdm(loader):
            x = x.cuda(); y = y.cuda() 

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_track.append(loss.item())

        # save model
        print(f'average loss: {np.mean(loss_track):.5f}')
        print(f'saving model')
        torch.save(model, f'swin_t.pt')

    return model

# orthonormalize the 2 vectors defining the hyperplane
# using gram-schmidt: (https://en.wikipedia.org/wiki/Gram-Schmidt_process)
def orthonormalize(v):
    v1, v2 = v
    v1 = v1 / torch.linalg.norm(v1)
    #v2 -= torch.einsum('c h w, c h w -> ', v1, v2) * v1
    v2 = v2 / torch.linalg.norm(v2)
    return torch.stack([v1, v2])

# NOTE: this is slow... too bad!!
# sample square grid of images on the hyperplane, spaced by eps
def make_grid(x, v, grid_size, grid_samples):
    eps = grid_size / grid_samples

    v1, v2 = v
    shape = (grid_samples, grid_samples, *x.shape[1:])
    grid = torch.zeros(shape).cuda()
    for i in range(grid_samples):
        for j in range(grid_samples):
            step_1 = (i - grid_samples//2) * eps * v1
            step_2 = (j - grid_samples//2) * eps * v2
            grid[i, j] = x + step_1 + step_2
    return grid

@ torch.no_grad()
def draw_frames(x, model, n_frames, grid_size, grid_samples):
    print('making frames...')

    # sample two random vectors in image-space
    # which define the hyperplane
    shape = (2, *x.shape[1:])
    v_init = torch.randn(shape).cuda()
    
    # sample a vector which defines the rotation
    c = 2 * np.pi * torch.rand(shape).cuda()

    # main loop to draw frames in gif
    frames = []
    T = np.linspace(0, np.pi, n_frames)

    for i, t in enumerate(tqdm(T)):
        # rotate the hyperplane
        v = torch.sin(t * c) + v_init
        v = orthonormalize(v)

        # sample grid of images on the hyperplane
        x_grid = make_grid(x, v, grid_size, grid_samples) 

        # batch forward pass by rows
        y_out = []
        for i in range(x_grid.shape[0]):
            y = model(x_grid[i]).argmax(dim=-1)
            y_out.append(y)
        y = torch.stack(y_out)

        frames.append(y.cpu())

    return torch.stack(frames)

if __name__ == '__main__':
    args = fetch_args()

    model = load_model(args.model_path).cuda()

    # train swin-t model on cifar-10
    if args.model_path is None:
        loader = fetch_loader(args.batch_size, args.num_workers, 'train')
        model = train_swin(model, loader)
    
    loader = fetch_loader(args.batch_size, args.num_workers, 'test')

    # get a single cifar-10 image incorrectly classified by the model
    # the image is the origin around which we 
    # will rotate a hyperplane in image-space
    while True:
        x, y = next(iter(loader))
        x = x.cuda(); y = y.cuda()

        pred = model(x).argmax(dim=-1)
        ids = torch.where(pred != y)[0]

        if len(ids) > 0:
            x = x[ids[0]][None, ...]
            break

    # main loop to draw frames for gif
    frames = draw_frames(x, model, args.n_frames, args.grid_size, args.grid_samples)
    make_gif(frames, args.save_path)
