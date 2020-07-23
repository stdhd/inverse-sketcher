from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold
import logging
import os
import torch
from PIL import Image
import torchvision
from tqdm import tqdm
import numpy as np
from random import uniform

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

    def __init__(self, root_dir, transform=None, return_path=False, only_classes=None, only_one_sample=False, noise_factor=0.002, load_on_request=False):
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
        self.load_on_request = load_on_request
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
            num_classes = 0
            inc = {}
            if self.only_classes:
                for n in self.only_classes:
                    inc[n] = False
            for classfolder in tqdm(folder_iterator, "Processing Sketch Metadata"):
                #label = self.class_numbers.get(classname)
                #if label is None:
                #    logger.error("Warning: Undefined class name {} in data directory {}".format(classname, self.__root_dir))
                if os.path.isdir(os.path.join(self.__sketch_dir, classfolder.name)) and (self.only_classes==None or classfolder.name in self.only_classes):
                    num_classes += 1
                    inc[classfolder.name] = True
                    with os.scandir(os.path.join(self.__sketch_dir, classfolder.name)) as sketch_iterator:
                        for file in sketch_iterator:
                            if not file.name.startswith(".") and not file.name.endswith(".svg"):
                                path_sketch = os.path.join(self.__sketch_dir, classfolder.name, file.name)
                                if "SketchyDatabase" in path_sketch:
                                    path_real = os.path.join(self.__real_dir, classfolder.name, file.name.split("-")[0] + ".jpg")
                                elif "ShoeV2_F" in path_sketch:
                                    path_real = os.path.join(self.__real_dir, classfolder.name, file.name.split("_")[0] + ".png")
                                elif "flickr" in path_sketch:
                                    path_real = os.path.join(self.__real_dir, classfolder.name, file.name.split("_")[0] + ".png")
                                else:
                                    raise(RuntimeError("Unknown dataset {}".format(self.__sketch_dir.split("/")[1])))
                                if not os.path.exists(path_real):
                                    logger.error("Warning: Could not find real image named {} corresponding to sketch {}".format(path_real, path_sketch))
                                    continue
                                if not self.load_on_request:
                                    image, sketch = Image.open(path_real), Image.open(path_sketch)
                                    if np.asarray(sketch).shape[-1] == 4:
                                        sketch = torchvision.transforms.ToPILImage()(torch.from_numpy(np.asarray(sketch)[:,:,-1]))
                                        sub = False
                                    else:
                                        sketch = sketch.convert("L")
                                        sub = True
                                        #sub = "SketchyDatabase" in path_sketch
                                    if not self.__transform is None:
                                        sketch = self.__transform(sketch)
                                        image = self.__transform(image)
                                    image = tensor_transform(image)
                                    sketch = tensor_transform(sketch)

                                    #Make the background pixels black and brushstroke pixels white
                                    if sub:
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
            for n in inc.keys():
                if not inc[n]:
                    print("Missing class {}".format(n))
            print("Training on {} classes".format(num_classes))
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

            path_sketch = meta.get_sketch()
            path_real = meta.get_real()

            # Please leave this here, as the dataset in my colab has some duplicates:
            if path_sketch.endswith(' (1).png'):
                path_sketch = path_sketch.split(" ")[0] + ".png"

            sketch = Image.open(path_sketch)
            if np.asarray(sketch).shape[-1] == 4:
                sketch = torchvision.transforms.ToPILImage()(torch.from_numpy(np.asarray(sketch)[:,:,-1]))
                sub = False
            else:
                sketch = sketch.convert("L")
                sub = True
                #sub = "SketchyDatabase" in path_sketch

            image = Image.open(path_real)

            if not self.__transform is None:
                sketch = self.__transform(sketch)
                image = self.__transform(image)

            tensor_transform = torchvision.transforms.ToTensor()
            image = tensor_transform(image)
            sketch = tensor_transform(sketch)

            #Make the background pixels black and brushstroke pixels white
            if sub:
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


class CompositeIterSingle():
    def __init__(self, loader1, loader2, p, epoch_len=2000):
        self.loader1 = iter(loader1)
        self.loader2 = iter(loader2)
        self.p = p
        self.epoch_len = epoch_len
        self.num_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.num_calls += 1
        if self.num_calls > self.epoch_len:
            raise StopIteration
        if uniform(0, 1) > self.p:
            return self.loader2.__next__()
        else:
            return self.loader1.__next__()

    def __len__(self):
        return self.epoch_len# + len(self.loader2)

class CompositeDataloader(object):
    def __init__(self, dataloader1, dataloader2, p=0.5, anneal_rate=1):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        assert anneal_rate >= 0 and anneal_rate <= 1, "please choose anneal betweem 0 and 1 and order datasets accordingly"
        self.anneal_rate = anneal_rate
        self.p = p
        self.iter = CompositeIterSingle(self.dataloader1, self.dataloader2, self.p, epoch_len=min(2000, len(self.dataloader1)))

    def __iter__(self):
        self.iter = CompositeIterSingle(self.dataloader1, self.dataloader2, self.p, epoch_len=min(2000, len(self.dataloader1)))
        return self.iter

    def __len__(self):
        return len(self.iter)

    def set_p(self, p):
        self.p = p
        self.iter.p = p

    def anneal_p(self):
        self.set_p(1 - (1 - self.p)*self.anneal_rate)
