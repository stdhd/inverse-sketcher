from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
import logging
import os
import torch
from PIL import Image
import torchvision
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageMetaData(object):

    def __init__(self, path_sketch, path_real, label, real=None, sketch=None):
        self.__label = label
        self.__path_sketch = path_sketch
        self.__path_real = path_real
        if not (real is None) and not (sketch is None):
            self.__real, self.__sketch = real, sketch

    def get_sketch(self):
        return self.__path_sketch

    def get_real(self):
        return self.__path_real

    def get_class(self):
        return self.__label

    def get_images(self):
        return self.__real, self.__sketch


class ImageDataSet(Dataset):

    def __init__(self, root_dir, transform=None, return_path=False, only_classes=None, only_one_sample=False, noise_factor=0.005, load_on_request=False):
        """
        root_dir: directory of the dataset
        include_unk: Whether to include the unknown class
        transform: transormations to be applied every time a batch is loaded
        only_classes: List of folder names to take data from, exclusively
        only_one_sample: If this is true, train and test set only contain ONE same sample
        noise_factor: Factor for the noise added to both sketch and image
        """
        self.__sketch_dir = os.path.join(root_dir,  "sketch")
        self.__real_dir = os.path.join(root_dir,  "photo")
        self.__transform = transform
        self.__meta = list()
        self.return_path = return_path
        self.only_classes = only_classes
        self.only_one_sample = only_one_sample
        self.noise_factor = noise_factor
        self.load_on_request = load_on_request
        self.get_class_numbers()
        self.__process_meta()

    def get_class_numbers(self):
        dict = {}
        with os.scandir(self.__sketch_dir) as folder_iterator:
            for i, classfolder in enumerate(folder_iterator):
                dict[classfolder.name] = i
            folder_iterator.close()
        self.__class_dict = dict

    def __process_meta(self):
        #class1, class2, ...
        tensor_transform = torchvision.transforms.ToTensor()
        with os.scandir(self.__sketch_dir) as folder_iterator:
            for classfolder in tqdm(folder_iterator, "Processing Sketch Metadata"):
                #label = self.class_numbers.get(classname)
                #if label is None:
                #    logger.error("Warning: Undefined class name {} in data directory {}".format(classname, self.__root_dir))
                if os.path.isdir(os.path.join(self.__sketch_dir, classfolder.name)) and (self.only_classes==None or classfolder.name in self.only_classes):
                    with os.scandir(os.path.join(self.__sketch_dir, classfolder.name)) as sketch_iterator:
                        for file in sketch_iterator:
                            if not file.name.startswith("."):
                                path_sketch = os.path.join(self.__sketch_dir, classfolder.name, file.name)
                                path_real = os.path.join(self.__real_dir, classfolder.name, file.name.split("-")[0] + ".jpg")
                                if not os.path.exists(path_real):
                                    logger.error("Warning: Could not find real image named {} corresponding to sketch {}".format(path_real, path_sketch))
                                    continue
                                if not self.load_on_request:
                                    image, sketch = Image.open(path_real), Image.open(path_sketch).convert("L")
                                    if not self.__transform is None:
                                        sketch = self.__transform(sketch)
                                        image = self.__transform(image)
                                    image = tensor_transform(image)
                                    sketch = tensor_transform(sketch)

                                    #Make the background pixels black and brushstroke pixels white
                                    sketch = (1 - sketch)
                                    image += self.noise_factor * torch.rand_like(image)
                                    sketch += self.noise_factor * torch.rand_like(sketch)
                                    self.__meta.append(ImageMetaData(path_sketch, path_real, self.__class_dict[classfolder.name], image, sketch))
                                    if self.only_one_sample:
                                        self.__meta.append(ImageMetaData(path_sketch, path_real, self.__class_dict[classfolder.name], image, sketch))
                                        sketch_iterator.close()
                                        folder_iterator.close()
                                        print("ONLY-ONE-SAMPLE-MODE (+ one duplicate to create split): Processed {} sketches".format(len(self.__meta)))
                                        return

                                else:
                                    self.__meta.append(ImageMetaData(path_sketch, path_real, self.__class_dict[classfolder.name]))
                                    if self.only_one_sample:
                                        self.__meta.append(ImageMetaData(path_sketch, path_real, self.__class_dict[classfolder.name]))
                                        sketch_iterator.close()
                                        folder_iterator.close()
                                        print("ONLY-ONE-SAMPLE-MODE (+ one duplicate to create split): Processed {} sketches".format(len(self.__meta)))
                                        return
                        sketch_iterator.close()
            print("Processed {} sketches".format(len(self.__meta)))
            folder_iterator.close()


    def __len__(self):
        return len(self.__meta)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.to(dtype=torch.int)
        meta = self.__meta[idx]

        if self.load_on_request:

            path_sketch = meta.get_sketch()
            path_real = meta.get_real()

            # Please leave this here, as the dataset in my colab has some duplicates:
            if path_sketch.endswith(' (1).png'):
                path_sketch = path_sketch.split(" ")[0] + ".png"

            sketch = Image.open(path_sketch).convert("L")
            image = Image.open(path_real)

            if not self.__transform is None:
                sketch = self.__transform(sketch)
                image = self.__transform(image)

            tensor_transform = torchvision.transforms.ToTensor()
            image = tensor_transform(image)
            sketch = tensor_transform(sketch)

            #Make the background pixels black and brushstroke pixels white
            sketch = (1 - sketch)

            # Add noise
            image += self.noise_factor * torch.rand_like(image)
            sketch += self.noise_factor * torch.rand_like(sketch)

            #trans = torchvision.transforms.ToPILImage()
            #trans(image).show()
        else:
            meta = self.__meta[idx]
            image, sketch = meta.get_images()

        return sketch, image, meta.get_class()
