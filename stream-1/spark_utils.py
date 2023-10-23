from ast import literal_eval
import os
import matplotlib.pyplot as plt
from skimage import io , img_as_uint 
import pandas as pd
import matplotlib.patches as mpatches


def process_labels(labels_dir,split):
    path = os.path.join(labels_dir, f"{split}.csv")
    labels = pd.read_csv(path)
    return  labels



class SPARKDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, class_map, root_dir='./data',split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.labels = process_labels(root_dir,split)
        self.class_map =  class_map

    def get_image(self, i=0):

        """ Loading image as PIL image. """

        sat_name = self.labels.iloc[i]['class']
        img_name = self.labels.iloc[i]['filename']
        image_name = f'{self.root_dir}/{img_name}'

        image = io.imread(image_name)

        return image , self.class_map[sat_name]

    def get_bbox(self, i=0):

        """ Getting bounding box for image. """

        bbox = self.labels.iloc[i]['bbox']
        bbox    = literal_eval(bbox)
        
        min_x, min_y, max_x, max_y = bbox

        return min_x, min_y, max_x, max_y 



    def visualize(self,i, size=(15,15),  ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
            
        image, img_class = self.get_image(i)
        min_x, min_y, max_x, max_y   = self.get_bbox(i)

        ax.imshow(image,vmin=0, vmax=255)


        rect = mpatches.Rectangle((min_y, min_x), max_y - min_y, max_x - min_x,
                                        fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        label = f"{list(self.class_map.keys())[list(self.class_map.values()).index(img_class)]}"
        
        ax.text(min_y, min_x-20, label,color='white',fontsize=15)
        ax.set_axis_off()

        return 

    
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    has_pytorch = True
    print('Found Pytorch')
except ImportError:
    has_pytorch = False

    
if has_pytorch:
    class PyTorchSparkDataset(Dataset):

        """ SPARK dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, class_map, split='train', root_dir='', transform=None,detection = True):

            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'validation', 'test'}:
                raise ValueError('Invalid split, has to be either \'train\', \'validation\' or \'test\'')


            self.class_map =  class_map
            
            self.detection = detection
            self.split = split 
            self.root_dir = os.path.join(root_dir, self.split)
            
            self.labels = process_labels(root_dir,split)
                
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            
            sat_name = self.labels.iloc[idx]['class']
            img_name = self.labels.iloc[idx]['filename']
            image_name = f'{self.root_dir}/{img_name}'
            
            image = io.imread(image_name)


            if self.transform is not None:
                torch_image = self.transform(image)
            
            else:
                torch_image = torch.from_numpy(image).permute(2,1,0)
                
            if self.detection:
                
                bbox = self.labels.iloc[idx]['bbox']
                bbox = literal_eval(bbox)

                return torch_image, self.class_map[sat_name] , torch.tensor(bbox)

            return torch_image, torch.tensor(self.class_map[sat_name])
else:
    class PyTorchSparkDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')
            
