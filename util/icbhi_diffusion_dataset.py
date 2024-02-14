from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,generate_spectrogram
from .augmentation import augment_raw_audio
import torchaudio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_meta_infor(real_or_gen, args):
    
    if args.meta_mode == 'none':
        meta_label = -1
    
    
    elif args.meta_mode == 'mixed':
        if real_or_gen == 'real':
            meta_label = 0
        else:
            meta_label = 1
    
                   
    return meta_label


def get_device_infor(hos_or_iphone, args):
    
    if args.device_mode == 'none':
        device_label = -1
    
    
    elif args.device_mode == 'mixed':
        if hos_or_iphone == 'hospital':
            device_label = 0
        else:
            device_label = 1
    
                   
    return device_label


def extract_features(audio_data):
    # Assuming audio_data is a raw audio waveform, convert it to a frequency representation, e.g., Mel-spectrogram
    # You can also extract other features like MFCCs, Zero Crossing Rate, etc.
    # Here's an example of converting to a Mel-spectrogram
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    mel_spec = mel_spec_transform(audio_data)
    return mel_spec.mean(dim=2) 



class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, real, print_flag=True):
        train_data_folder = os.path.join(args.data_folder, 'training', 'real') if real else os.path.join(args.data_folder, 'training', args.real_gen_dir)
        test_data_folder = os.path.join(args.data_folder, 'test', 'real')
        self.train_flag = train_flag
        
        
        
        
        if self.train_flag:
            self.data_folder = train_data_folder
        else:
            self.data_folder = test_data_folder
        self.transform = transform
        self.args = args
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        
        self.data_glob = sorted(glob(self.data_folder+'/*/*.wav')) if not real else sorted(glob(self.data_folder+'/*.wav'))
        

                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        self.audio_images_2 = []
        self.features = []
        self.patient_ids =[]
        self.domains = []
        self.classes = []
  
            
        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            

   
            patient_id, _, domain_class = file_id.split('-')
            domain, class_label = domain_class[0], domain_class[1]
            patient_id = patient_id.split(')')[0]
            self.patient_ids.append(patient_id)
            self.domains.append(domain)
            self.classes.append(class_label)
            
            
            audio, sr = torchaudio.load(index)
            audio_feature = extract_features(audio)
            self.features.append(audio_feature.numpy())

            
            file_id_tmp = file_id.split('.wav')[0]
            # stetho = file_id_tmp.split('_')[4]
            if 'index' in file_id_tmp:
                label = file_id_tmp.split('_')[-2]
                real_or_gen = 'gen'
                meta_label = get_meta_infor(real_or_gen, self.args)
                hos_or_iphone = 'hospital'
                device_label = get_device_infor(hos_or_iphone, self.args)
                print("wrong")
            else:
                # label = file_id_tmp.split('-')[2]
                device_label = 0
                # for total
                label = file_id_tmp.split('-')[2][1]
                device_label = file_id_tmp.split('-')[2][0]
                patient_label = file_id_tmp.split(')')[0]
                if label == "N":
                    label = 0
                elif label == "B":
                    label = 1
                elif label == "C":
                    label = 1
                elif label == "W":
                    label = 1
                else:
                    print(index)
                    continue
                real_or_gen = 'real'
                if device_label == 'H':
                    hos_or_iphone = 'hospital'
                else:
                    hos_or_iphone = 'iphone'
                device_label = get_device_infor(hos_or_iphone,self.args)

            meta_label = get_meta_infor(real_or_gen, self.args)
            device_label = get_device_infor(hos_or_iphone,self.args)
            audio, sr = torchaudio.load(index)
            
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)                
                    
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels)
                    audio_image.append(image)
                else:
                    image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels) 
                    # print("image shape : ", image.shape)
                    audio_image.append(image)
            
            if self.args.meta_mode == 'none' and self.args.device_mode =='none':
                self.audio_images.append((audio_image, int(label)))
            elif self.args.meta_mode == 'mixed' and self.args.device_mode =='none':
                self.audio_images.append((audio_image, int(label), meta_label)) 
            elif self.args.meta_mode == 'none' and self.args.device_mode =='mixed'and self.args.method != 'pdc':
                self.audio_images.append((audio_image, int(label), device_label)) 
            elif self.args.method == 'pdc':
                self.audio_images.append((audio_image, int(label), device_label,int(patient_label))) 
            else:
                self.audio_images.append((audio_image, int(label), meta_label,device_label)) 
                
            self.audio_images_2.append((audio_image, int(label),index))
            
      
        # ==========================================================================

   


    def __getitem__(self, index):
        if self.args.meta_mode == 'none' and self.args.device_mode == 'none':
            audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        elif self.args.meta_mode == 'mixed' and self.args.device_mode == 'none':
            audio_images, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
        elif self.args.meta_mode == 'none' and self.args.device_mode == 'mixed' and self.args.method != 'pdc':
            audio_images, label, device_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
        elif self.args.method == 'pdc':
            audio_images, label, device_label ,patient_label= self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2],self.audio_images[index][3]
        else:
            audio_images, label, meta_label,device_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2],self.audio_images[index][3]

        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.train_flag:
            if self.args.meta_mode == 'none' and self.args.device_mode == 'none':
                return audio_image, torch.tensor(label)
            elif self.args.meta_mode == 'mixed' and self.args.device_mode == 'none':
                return audio_image, (torch.tensor(label), torch.tensor(meta_label))
            elif self.args.meta_mode == 'none' and self.args.device_mode == 'mixed' and self.args.method != 'pdc':
                return audio_image, (torch.tensor(label), torch.tensor(device_label))
            elif self.args.cluster_mode == 'mixed':
                cluster_label = self.cluster_labels[index]
                return audio_image, (torch.tensor(label), torch.tensor(cluster_label))
            elif self.args.method == 'pdc':
                return audio_image, (torch.tensor(label), torch.tensor(device_label),torch.tensor(patient_label))
            else:
                return audio_image, (torch.tensor(label), torch.tensor(meta_label),torch.tensor(device_label))
        else:
            # for visualization
            return audio_image, (torch.tensor(label), torch.tensor(device_label),torch.tensor(patient_label))
            # return audio_image, torch.tensor(label)
        
        

    def __len__(self):
        return len(self.data_glob)