import os, sys, argparse, torch, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from einops import rearrange, repeat
from PIL import Image

from model import CrappyNet
from dataload import Dataloader, gen_dataset, load_iid, load_ood

# sorts keys according to alpha numeric
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# saves images into a gif
def gif_save(path, name):
    pull_path = os.path.join(path)
    ims = os.listdir(pull_path)   
    ims = sorted(ims, key=natural_key)

    # read files
    video = []
    for i, f in enumerate(ims):
        file = os.path.join(pull_path, f)
        frame = Image.open(file)
        video.append(frame)
        

    # save gif
    save_path = os.path.join(path, '..', name+'.gif') 
    video[0].save(save_path, format='gif',
                   append_images=video[1:],
                   save_all=True,
                   duration=60, loop=0)   
    return

# get on image from each class
def pick_data(x, y, n_classes=10):
    s = x.shape
    out = torch.empty((n_classes, *s[1:]))
    for i in range(n_classes):
        opt = torch.where(y == i)[0] # all inds where class is i
        ind = np.random.randint(opt.shape[0]) # select a random one
        out[i] = x[opt[ind]]
    return out

# return batch of images based on x 
# this function makes rows for the visualization
def make_row(x, r, v1, v2, w, eps, device):
    eps_vec = torch.zeros(w+1, *x.shape).to(device)
    for i in range(w+1): # this is slow
        q1 = eps * (i - w//2) * v1
        q2 = eps * (r - w//2) * v2
        eps_vec[i,:,:,:] = q1 + q2 
    x = repeat(x, 'c h w -> k c h w', k=w+1)
    return x + eps_vec

def norm(x):
    return x / torch.linalg.norm(x)

# select model to load
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', default=None, type=str, help='name of experiment (will load that model)')
    return parser.parse_args()

if __name__ == '__main__':
    # get args and perform checks
    args = get_args()
    assert args.name is not None

    # load model
    path = os.path.join('results', args.name)
    model = torch.load(os.path.join(path, 'best_ood', 'model_ood.pt'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print('using: ', device)

    # load data
    data_path = 'data/mnist/'
    x_ood, y_ood = load_ood(data_path, device)

    # push to device
    model = model.to(device)

    # settings for vis
    w = 400 # width of window
    q = np.sqrt(28**2)
    eps = 2 * q/w # diff per change in pixel
    print(eps)

    path = os.path.join(path, 'imgs')
    os.makedirs(path, exist_ok=True)
        
    # main video gen loop
    c_ = 4
    while True:
        ind = np.random.randint(x_ood.shape[0])
        x = x_ood[ind,:,:,:]
        y = model(x).argmax().item()
        print(y)
        if y == c_: break

    # sample initial hyper-plane vectors
    s = np.prod([a for a in x.shape])
    a = torch.randn(s)
    b = torch.randn(s)

    # make orthogonal and normalize
    a0 = norm(a)
    b0 = norm(b)

    # sample frequency vectors
    c = 2*np.pi * torch.rand(2, s)
    
    # for images
    x_input = torch.empty((w+1, w+1, *x.shape))
    img = torch.zeros((w+1, w+1))

    # main video gen
    patches = 4
    mod_k = (w+1) // patches
    T = np.linspace(0, 2*np.pi, 100)
    for i, t in enumerate(T):
        img, x_patch = None, None
        a = torch.sin(t + c[0]) + a0 
        b = torch.sin(t + c[1]) + b0 
        a = norm(a)
        b = norm(b)
        
        # make image loop
        for j in range(w+1): # center pixel to be initial guess
            x_row = make_row(x, j, a.view(*x.shape), b.view(*x.shape), w, eps, device)
            x_patch = torch.cat((x_patch, x_row)) if x_patch is not None else x_row 

            if j % mod_k == 0 and j != 0 or j == w:
                out = model(x_patch).exp()
                out = out.argmax(dim=1).cpu()
                img = torch.cat((img, out)) if img is not None else out
                x_patch = None
        img = img.view((w+1, w+1))

        # plot
        plt.imshow(img, vmin=0, vmax=9, cmap=colormaps['terrain'])
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'vis_{}.png'.format(i)))
        plt.close()

    gif_save(os.path.join(path), 'class_{}'.format(c_))

