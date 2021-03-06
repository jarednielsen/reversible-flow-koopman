from libs.args.args import argchoice
import numpy as np
import torch
from torchvision import datasets, transforms
import os

from glob import glob
import libs.args.args as args
import sys
import os
sys.path.append('./libs/moving-symbols/moving_symbols')
from moving_symbols import MovingSymbolsEnvironment

# class MnistDataset(datasets.MNIST):
#   def __init__(self, root='/tmp/mnist', train=True, download=True, overfit=False):
#   	self.overfit = overfit
#   	super(MnistDataset, self).__init__(
#   		root, 
#   		download=download, 
#       train=train,
#   		transform=transforms.Compose([
#                            transforms.Resize((32,32)),
#                            transforms.ToTensor(),
#                            lambda x: torch.cat([x, x[0:1]], dim=0),
#                            #lambda x: x + torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x) * (1. / 256.))
#                            lambda x: x + torch.zeros_like(x).uniform_(0., 1./ 255.),
#                            #transforms.Normalize((0.1715,), (1.0,))
#                        ]))

#   def __len__(self):
#   	if self.overfit:
#   		return 32
#   	else:
#   		return super(Dataset, self).__len__()

class GolfSwingClips(torch.utils.data.Dataset):
    """
    Returns (N, 3, 32, 32) tensor by default for a single item.
    Compare to (32, 32, 3 * 11) tensor in AVG dataset.
    """

    def __init__(self, root="/mnt/pccfs/backed_up/jaredtn/data/ucf_action_combined/.Clips/", sequence_length=10, num_channels=3,
                transformation=transforms.Compose([
                  transforms.Resize((32,32)),
                  transforms.ToTensor(),
                ]),
                projection_dimensions=32, use_pca=False, overfit=False, repeats=100, train=False):
      super().__init__()
      self.root = root
      self.n_clips = len(glob(root + '*'))

    def denormalize(self, x):
      return 255/2 * (x + 1)

    def __getitem__(self, index):
      path = self.root + str(np.random.choice(self.n_clips)) + '.npz'
      clip = np.load(path)['arr_0'] # (H, W, 3 * FRAMES) concatenates along the RGB axis. Undo this.
      h, w, threef = clip.shape
      f = threef // 3
      reshaped_clip = np.zeros((f, 3, h, w))
      for i in range(f):
        reshaped_clip[i] = clip[:,:,3*i:3*(i+1)].transpose([2, 0, 1])
      reshaped_clip = torch.Tensor(reshaped_clip)

      reshaped_clip = np.transpose(clip, [2, 0, 1])
      reshaped_clip = 2 / 255 * reshaped_clip - 1 # normalize

      return reshaped_clip, reshaped_clip[0][0]

    def __len__(self):
      return self.n_clips

    def __repr__(self):
      summary =   '                   Name: {}\n'.format(self.__class__.__name__)
      summary += '                    Size: {}\n'.format(self.__getitem__(0)[0].shape)
      summary += '               Min Value: {:.2f}\n'.format(self.__getitem__(0)[0].min().item())
      summary += '               Max Value: {:.2f}\n'.format(self.__getitem__(0)[0].max().item())

      return summary 

class BouncingMNIST(torch.utils.data.Dataset):
    """
    Returns (10, 3, 32, 32) tensor by default for a single item.
    Compare to (32, 32, 3 * 11) tensor in AVG dataset.
    """
    def __init__(self, root="/mnt/pccfs/backed_up/jaredtn/data/bouncing_mnist/Test_Single/", sequence_length=15, num_channels=3,
                projection_dimensions=32, use_pca=False, overfit=False, repeats=100, train=False, height=32, width=32):
      super().__init__()

      transformation=transforms.Compose([
        transforms.Resize((height,width)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
      ])
      
      # self.repeats = repeats
      # self.overfit = overfit
      # if overfit:
      #   self.size = args.reader().batch_size
      
      # Set up image loader
      self.dataset_folder = datasets.ImageFolder(os.path.join(root), transform = transformation)        
      self.total_num_frames = len(self.dataset_folder)
      self.sequence_length = sequence_length

      c, h, w = self.dataset_folder[0][0].size()
      
      # Create data, which is a tensor containing each image sequentially. Length is the number of files in the folder
      # c + (c % 2)
      self.data = torch.zeros([self.total_num_frames, c, h, w]).float()
      for i in range(self.total_num_frames):
          self.data[i] = self.dataset_folder[i][0]

      self.centering_const = self.data.view(self.total_num_frames, -1).mean()
      self.normalizing_const = (self.data - self.centering_const).abs().mean()

    def normalize(self, x):
      return (x - self.centering_const) / self.normalizing_const

    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self,index):
      assert index < self.__len__(), "Index {} is out-of-bounds for dataset with length {}".format(index, self.__len__())
      # Ensure we start at the beginning of a sequence
      frames = self.data[index * self.sequence_length : (index + 1) * self.sequence_length]
      # Add Gaussian noise
      frames += (torch.rand_like(frames) - .5) * 1 / 255
      frames = self.normalize(frames)

      if len(frames) == 0:
        import pdb; pdb.set_trace()
      return frames, frames[0,0]

    def __len__(self):
      return self.total_num_frames // self.sequence_length

    def __repr__(self):
      summary =   '                   Name: {}\n'.format(self.__class__.__name__)
      summary += '                    Size: {}\n'.format(self.__getitem__(0)[0].shape)
      summary += '               Min Value: {:.2f}\n'.format(self.__getitem__(0)[0].min().item())
      summary += '               Max Value: {:.2f}\n'.format(self.__getitem__(0)[0].max().item())

      return summary  

class GolfSwing(torch.utils.data.Dataset):
    """
    Returns (10, 3, 32, 32) tensor by default for a single item.
    Compare to (32, 32, 3 * 11) tensor in AVG dataset.
    """
    def __init__(self, root="/mnt/pccfs/backed_up/jaredtn/data/ucf_action_single_clip/Train/", sequence_length=15, num_channels=3,
                transformation=transforms.Compose([
                  transforms.Resize((32,32)),
                  transforms.ToTensor(),
                ]),
                projection_dimensions=32, use_pca=False, overfit=False, repeats=100, train=False):
      super().__init__()
      
      self.repeats = repeats
      self.overfit = overfit
      if overfit:
        self.size = args.reader().batch_size
      
      # Set up image loader
      self.dataset_folder = datasets.ImageFolder(os.path.join(root) ,transform = transformation)        
      self.total_num_frames = len(self.dataset_folder)
      self.sequence_length = sequence_length

      c, h, w = self.dataset_folder[0][0].size()
      
      # Create data, which is a tensor containing each image sequentially. Length is the number of files in the folder
      # c + (c % 2)
      self.data = torch.zeros([self.total_num_frames, c, h, w]).float()
      for i in range(self.total_num_frames):
          self.data[i] = self.dataset_folder[i][0]

      self.centering_const = 0 #self.data.view(self.total_num_frames, -1).mean()
      self.normalizing_const = 0.1 #(self.data - self.centering_const).abs().mean()


    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self,index):
      index = index % ((self.total_num_frames - self.sequence_length + 1) if not self.overfit else self.size)
      raw = self.data[index: index + self.sequence_length]
      raw += (torch.rand_like(raw) - .5) * 1 / 256. 
      raw = (raw - self.centering_const) / self.normalizing_const

      return raw, raw[0,0]

    def __len__(self):
      if self.overfit:
        # Total number of sequences = Total Frames - Sequence Length + 1
        return self.size * self.repeats

      return (self.total_num_frames - self.sequence_length + 1) * self.repeats

    def __repr__(self):
      summary =   '                   Name: {}\n'.format(self.__class__.__name__)
      summary += '                    Size: {}\n'.format(self.__getitem__(0)[0].shape)
      summary += '               Min Value: {:.2f}\n'.format(self.__getitem__(0)[0].min().item())
      summary += '               Max Value: {:.2f}\n'.format(self.__getitem__(0)[0].max().item())

      return summary            

class RotatingCube(torch.utils.data.Dataset):
    def __init__(self, root="/mnt/pccfs/not_backed_up/data/cube_data/spherecube_const_pitch_yaw_16", sequence_length=10, num_channels=3, 
                  transformation=transforms.Compose([transforms.ToTensor()]), 
                  projection_dimensions=2 * 8 * 8, use_pca=False, overfit=False, repeats=1, train=False):
      super(RotatingCube, self).__init__()
      
      self.repeats = repeats
      self.overfit = overfit
      if overfit:
        self.size = args.reader().batch_size
      
      # Set up image loader
      self.dataset_folder = datasets.ImageFolder(os.path.join(root) ,transform = transformation)        
      self.total_num_frames = len(self.dataset_folder)
      self.sequence_length = sequence_length
      self.use_pca = use_pca

      c, h, w = self.dataset_folder[0][0].size()
      
      # Create data, which is a tensor containing each image sequentially. Length is the number of files in the folder
      # c + (c % 2)
      self.data = torch.zeros([self.total_num_frames, c, h, w]).float()
      for i in range(self.total_num_frames):
          self.data[i] = self.dataset_folder[i][0]
      if use_pca:
        two_d = self.data.view(self.total_num_frames, -1)

        assert self.total_num_frames > projection_dimensions, f"Number of projection dimensions ({projection_dimensions}) is greater than number of samples ({self.total_num_frames}), using number of samples"
        assert projection_dimensions % 2 == 0, 'projection_dimensions must be even'

        # A PCA Method could be included here if desired. 
        # self.data = PCA(self.data)
        # preprocess the data
        X_mean = torch.mean(two_d,0)
        two_d = two_d - X_mean.expand_as(two_d)
        self.U, self.S, self.V = torch.svd(torch.t(two_d), some=True)
        self.data = torch.mm(two_d, self.U[:,:projection_dimensions]) # k is num dimensions
        self.projection_dimensions = projection_dimensions

      #self.data += (torch.rand_like(self.data) - .5) * 1 / 256

      self.centering_const = self.data.view(self.total_num_frames, -1).mean()
      self.normalizing_const = (self.data - self.centering_const).abs().mean()

      #self.data = torch.randn_like(self.data)

    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self,index):
      index = index % ((self.total_num_frames - self.sequence_length + 1) if not self.overfit else self.size)
      raw = self.data[index: index + self.sequence_length]
      raw += (torch.rand_like(raw) - .5) * 1 / 256.
      raw = (raw - self.centering_const) / self.normalizing_const

      if self.use_pca:
        k = int(np.sqrt(raw.size(1) / 2))
        return raw.view(self.sequence_length, 2, k, -1), raw[0,0]
      else:
        return raw, raw[0,0]

    def __len__(self):
      if self.overfit:
        # Total number of sequences = Total Frames - Sequence Length + 1
        return self.size * self.repeats

      return (self.total_num_frames - self.sequence_length + 1) * self.repeats

    def __repr__(self):
      summary =   '                   Name: {}\n'.format(self.__class__.__name__)
      summary += '                    Size: {}\n'.format(self.__getitem__(0)[0].shape)
      summary += '               Min Value: {:.2f}\n'.format(self.__getitem__(0)[0].min().item())
      summary += '               Max Value: {:.2f}\n'.format(self.__getitem__(0)[0].max().item())

      if self.use_pca:
        summary += '    Projection Dimension: {}\n'.format(self.projection_dimensions)
        summary += '                 Max SVs: {:.2f}\n'.format(self.S[:self.projection_dimensions].max().item())
        summary += '                 Min SVs: {:.2f}\n'.format(self.S[:self.projection_dimensions].min().item())

      return summary


class MovingSymbols(torch.utils.data.Dataset):
    raw = None
    def __init__(self, root="./libs/moving-symbols/data", sequence_length=10, 
                  overfit=False, size=20000, train=False, height=16, width=16, velocity=3):
      super(MovingSymbols, self).__init__()

      self.overfit = overfit
      self.batch_size = 16 #args.reader().batch_size
      self.sequence_length = sequence_length

      self.params = {
        'data_dir': os.path.join(root, 'mnist'),
        'split': 'training',
        'color_output': True,
        'symbol_labels': range(10),
        'position_speed_limits': (velocity, velocity),
        'video_size': (height, width),
        'scale_limits': (.4, .4) if height < 32 or width < 32 else (1, 1)
      }

      self.size = size if not self.overfit else self.batch_size
      print("size: {}".format(self.size))

      # save_location = os.path.join(root, 'moving_symbols_{}.{}.{}.{}.npy'.format(size, height, width, sequence_length))
      # if os.path.isfile(save_location):
      #   raw = np.load(save_location)
      #   print('Loaded')
      # else:
      #   raw = []
      #   from tqdm import tqdm
      #   for s in tqdm(range(self.size), desc='Building Dataset', leave=False):
      #     env = MovingSymbolsEnvironment(self.params, np.random.randint(0, 1))
      #     raw.append(np.array([np.asarray(env.next()) for _ in range(sequence_length)]) / 255.)
      #   raw = np.array(raw)
      #   if raw.nbytes < 1e10:
      #     print('Attempting to save {} bytes'.format(raw.nbytes))
      #     np.save(save_location, raw)
      #     print('Saved')
      #   else:
      #     print(raw.nbytes, 'is too big to save')

      # self.data = torch.from_numpy(raw.astype(np.float32)).permute(0, 1, 4, 2, 3).contiguous()
      
      self.centering_const = 0.     #self.data.mean()
      self.normalizing_const = .1 #self.data.abs().mean()

    def denormalize(self, x):
      return x * self.normalizing_const + self.centering_const

    def __getitem__(self, index):
      # index = index % (self.size if not self.overfit else self.batch_size)

      # raw = self.data[index]
      env = MovingSymbolsEnvironment(self.params, np.random.randint(0, 1))
      raw = np.array([np.asarray(env.next()) for _ in range(self.sequence_length)]) / 255.
      raw = torch.from_numpy(raw.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
      raw += (torch.rand_like(raw) - .5) * 1 / 255.
      raw = (raw - self.centering_const) / self.normalizing_const

      return raw, raw.view(-1)[0]

    def __len__(self):
      return self.size if not self.overfit else int(1e5)

    def __repr__(self):
      return ""

# class LTI2DSequence(torch.utils.data.Dataset):
#   def __init__(self, train=True, size=int(1e5), overfit=False, sequence_length=4, channels=4, state_dim=64 + 15, observation_dim=64):
#     self.__dict__.update(locals())

#     if overfit:
#       self.size = args.reader().batch_size

#     assert np.sqrt(observation_dim).is_integer(), 'observation_dim must be a perfect square. try {}'.format((int(np.sqrt(observation_dim)))**2)

#     rand_state = np.random.get_state()
#     np.random.seed(41)
#     self.state_dim = max(observation_dim, state_dim) * channels
#     self.observation_dim = observation_dim * channels
    
#     assert self.observation_dim <= self.state_dim, 'state_dim is too small'
    
#     self.A = np.random.randn(self.state_dim, self.state_dim)
#     u, s, v = np.linalg.svd(self.A)
#     self.A = u.dot(np.diag(1 + 0 * np.clip(.35 + np.random.rand(*s.shape), 0, 1))).dot(v)
#     self.A = np.eye(self.state_dim) * 1
#     u, s, v = np.linalg.svd(self.A)

#     self.C = np.zeros([self.observation_dim, self.state_dim])
#     self.C[:self.observation_dim, :self.observation_dim] = np.eye(self.observation_dim)
#     self.x0 = np.random.randn(size, self.state_dim)

#     # Normalize to unit length
#     self.x0 /= np.sqrt((self.x0 ** 2).sum(axis=1, keepdims=True))

#     # self.state_noise = np.random.randn(self.sequence_length, self.state_dim)

#     np.random.set_state(rand_state)

#   def __getitem__(self, idx):
#     obs = np.empty([self.sequence_length, self.observation_dim])
#     x = self.x0[idx]
#     for t in range(self.sequence_length):
#       x = self.A.dot(x)
#       y = self.C.dot(x)
#       obs[t] = y

#     k = int(np.sqrt(self.observation_dim / self.channels))
#     obs = obs.reshape(self.sequence_length, self.channels, k, k).astype(np.float32)

#     return torch.from_numpy(obs), torch.from_numpy(obs[-1])

#   def __len__(self):
#     return self.size
