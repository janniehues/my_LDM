from omegaconf import OmegaConf
import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf

from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from ldm.util import instantiate_from_config
from taming.data.imagenet import retrieve
from taming.data.base import ImagePaths

class CondFromDirDataBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        # if False we skip loading & processing images and self.data contains filepaths
        # should I just remove this here?
        # self.process_images = True
        # Should we change this back? Which one 
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]



    def _prepare(self):
        raise NotImplementedError()

    # takes out images in ignore set
    def _filter_relpaths(self, relpaths):
        ignore = set([
            "n06596364_9591.JPEG",
        ])
        relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
        return relpaths

    def _load(self):
        # txt_filelist contains all path to files
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print("Removed {} files from filelist during filtering.".format(l1 - len(self.relpaths)))
        
        # synset is category name
        self.synsets = [p.split("/")[0] for p in self.relpaths]
        # datadir is head of data directory
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        # class dict name to id
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        
        self.class_labels = [class_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.synsets),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            # Resizing is done within ImagePaths using albumentations.SmallestMaxSize
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   tform=self.tforms,
                                   )
        else:
            self.data = self.abspaths
import random
class CondFromDirDataTrain(CondFromDirDataBase):
    NAME="CustomDataTrain"
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    # TODO put in ImagePath 
    def __getitem__(self, i):
        data=self.data[i]
        if self.uncond_prob>0:
            if random.random()<self.uncond_prob:
                data['class_label']=self.n_classes-1
        return data
    
    def _prepare(self):
        # datadir is the data root directory
        self.datadir=retrieve(self.config, "datadir", default=None)
        # is file list, this file is created below
        self.txt_filelist = os.path.join(self.datadir, "filelist.txt")
        self.random_crop = retrieve(self.config, "random_crop",
                                    default=False)
        self.uncond_prob=retrieve(self.config, "uncond_prob",default=0)
        if self.uncond_prob>0:
            self.n_classes=retrieve(self.config,"n_classes",default=None)
            self.n_classes=np.int64(self.n_classes)
            assert self.n_classes, "Please provide n_classes when using uncond_prob"
        self.process_images=retrieve(self.config, "process_images",
         default=True)
        # transforms
        image_transforms=retrieve(self.config,"image_transforms",default=False) 
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        self.tforms = image_transforms


        if not tdu.is_prepared(self.datadir):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.datadir))

            datadir = self.datadir
            filelist = glob.glob(os.path.join(datadir, "**", "*.jpg"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            tdu.mark_prepared(self.datadir)

