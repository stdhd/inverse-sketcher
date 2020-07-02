from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
import logging
import os
import torch
from PIL import Image
import torchvision

logger = logging.getLogger(__name__)


class ImageMetaData(object):

    def __init__(self, path_sketch, path_real, label):
        self.__label = label
        self.__path_sketch = path_sketch
        self.__path_real = path_real

    def get_sketch(self):
        return self.__path_sketch

    def get_real(self):
        return self.__path_real

    def get_class(self):
        return self.__label


class ImageDataSet(Dataset):

    def __init__(self, root_dir, transform=None, return_path=False, only_classes=None):
        """
        root_dir: directory of the dataset
        include_unk: Whether to include the unknown class
        transform: transormations to be applied every time a batch is loaded
        only_classes: List of folder names to take data from, exclusively
        """
        self.__sketch_dir = os.path.join(root_dir,  "sketch")
        self.__real_dir = os.path.join(root_dir,  "photo")
        self.__transform = transform
        self.__meta = list()
        self.return_path = return_path
        self.only_classes = only_classes

        self.get_class_numbers()
        self.__process_meta()

    def get_class_numbers(self):
        dict = {}
        for i, classname in enumerate(os.listdir(self.__sketch_dir)):
            dict[classname] = i
        self.__class_dict = dict

    def __process_meta(self):
        #class1, class2, ...
        for classname in os.listdir(self.__sketch_dir):
            #label = self.class_numbers.get(classname)
            #if label is None:
            #    logger.error("Warning: Undefined class name {} in data directory {}".format(classname, self.__root_dir))
            if os.path.isdir(os.path.join(self.__sketch_dir, classname)) and (self.only_classes==None or classname in self.only_classes):
                for filename in os.listdir(os.path.join(self.__sketch_dir, classname)):
                    if not filename.startswith("."):
                        path_sketch = os.path.join(self.__sketch_dir, classname, filename)
                        path_real = os.path.join(self.__real_dir, classname, filename.split("-")[0] + ".jpg")
                        if not os.path.exists(path_real):
                            logger.error("Warning: Could not find real image named {} corresponding to sketch {}".format(path_real, path_sketch))
                            continue
                        #if not self.load_on_request:
                        #    image = torch.from_numpy(cv2.imread(path))
                        self.__meta.append(ImageMetaData(path_sketch, path_real, self.__class_dict[classname]))
        print("Processed {} sketches".format(len(self.__meta)))


    def __len__(self):
        return len(self.__meta)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.to(dtype=torch.int)
        meta = self.__meta[idx]

        path_sketch = meta.get_sketch()
        path_real = meta.get_real()

        sketch = Image.open(path_sketch).convert("L")
        image = Image.open(path_real)

        if self.__transform is not None:
            sketch = self.__transform(sketch)
            image = self.__transform(image)

        tensor_transform = torchvision.transforms.ToTensor()
        image = tensor_transform(image)
        sketch = tensor_transform(sketch)
        #Make the background pixels black and brushstroke pixels white
        sketch = (1 - sketch)
        return sketch, image, meta.get_class()
