import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
from huggingface_hub import login
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2

class ROIDataset(Dataset):
    def __init__(self, img_list, transform):
        super().__init__()
        self.images_lst = img_list
        self.transform = transform

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


class UNIExtractor:
    def __init__(self, batch_size=512, device=None):
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        login('The login you need to apply for')  # token

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )
        self.model.eval().to(self.device)

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def crop_image(self, img, x, y, crop_size):
        left = x - int(crop_size // 2)
        top = y - int(crop_size // 2)
        right = left + crop_size
        bottom = top + crop_size
        return img[top:bottom, left:right]

    def extract(self, img_path, spatial_coords, crop_size=300):
        img = cv2.imread(img_path)
        img = np.array(img)

        sub_images = [
            self.crop_image(img, int(x), int(y), crop_size)
            for x, y in spatial_coords
        ]

        dataset = ROIDataset(sub_images, self.transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        feature_embs = []

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device)
                emb = self.model(batch)
                feature_embs.append(emb.cpu().numpy())

        feature_embs = np.concatenate(feature_embs, axis=0)
        return feature_embs