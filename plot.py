from netCDF4 import Dataset
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import xarray as xr

file_path = "D:\WRFOUT_dataprocess\EXTRACT"
files = os.listdir(f'{file_path}')

# for file in files:
#    data__ = xr.open_dataset(f'{file_path}/{file}')
#
#    var = data__.variables["wind_speed"].squeeze()
#    # print(var.data.shape[0], var.data.shape[1])
#    # plt.figure(figsize=(var.data.shape[0], var.data.shape[1]))
#    plt.imshow(var[:], interpolation=None, cmap='gray')
#    plt.axis('off')
#    plt.show()
#    str = file.split('.')
#    plt.savefig(f'train/{str[0]}.png', bbox_inches='tight',pad_inches=0)
#    plt.clf()
#    print(f"save successfully:{file}")

# class ncPlt():
#    """
#       :argument is file or tensor
#    """
#    def __init__(self, file=None, tensor=None):
#       self.file = file
#       self.tensor = tensor
#       self.data = xr.open_dataset(f'{file}')

def plot_grey(file=None, tensor=None, string=None):
   if(file is not None):
      data__ = xr.open_dataset(f'{file_path}/{file}')
      var = data__.data.variables["wind_speed"].squeeze()
      plt.imshow(var[:], interpolation='nearest', cmap='gray')
      plt.axis('off')
      str = file.split('.')
      plt.savefig(f'results/train/{str[0]}.png', bbox_inches='tight', pad_inches=0)
      plt.clf()
      print(f"save successfully!")
   if(tensor is not None):
      plt.imshow(tensor[:], interpolation='nearest', cmap='gray')
      plt.axis('off')
      # string = file.split('.')
      plt.savefig(f'results/sample/{string}.png', bbox_inches='tight', pad_inches=0)
      plt.clf()
      print(f"save successfully!")

def plot_bicubic_gray(file=None, tensor=None):
   if (file is not None):
      var = file.data.variables["wind_speed"].squeeze()
      plt.imshow(var[:], interpolation='bicubic', cmap='gray')
      plt.axis('off')
      str = file.split('.')
      plt.savefig(f'results/bicubic/{str[0]}.png', bbox_inches='tight', pad_inches=0)
      plt.clf()
      print(f"save successfully!")
   if (tensor is not None):
      plt.imshow(tensor[:], interpolation='bicubic', cmap='gray')
      plt.axis('off')
      str = file.split('.')
      plt.savefig(f'results/bicubic/{str[0]}.png', bbox_inches='tight', pad_inches=0)
      plt.clf()
      print(f"save successfully!")