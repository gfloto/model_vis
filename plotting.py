import os
import matplotlib.pyplot as plt 

def make_gif(frames, save_path, cmap='terrain'):
    '''
    save a gif from a tensor of frames : [n_frames, height, width]
    assumes imagemagick is installed (uses ImageMagick's convert command)
    '''

    print('making gif...')
    os.makedirs('temp', exist_ok=False)

    for i, frame in enumerate(frames):
        plt.imshow(frame, cmap=cmap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('temp/frame_{}.png'.format(str(i).zfill(4)))
        plt.close()

    os.system(f'convert -delay 10 temp/*.png {save_path}')
    os.system('rm -rf temp')

