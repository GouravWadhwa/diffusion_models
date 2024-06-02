import torch

import blobfile as bf
import numpy as np

from PIL import Image

def load_data(data_params):
    shuffle = data_params["shuffle"]
    batch_size = data_params["batch_size"]

    dataset = ImageDataset(data_params=data_params)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    return iter(loader)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_params):
        super().__init__()

        self.resolution = data_params["resolution"]
        self.image_directory = data_params["image_directory"]

        self.num_classes = data_params["num_classes"]

        self.image_paths = []
        self.class_names = []

        self._list_image_paths()
        self.sorted_classes = {x: i for i, x in enumerate(sorted(self.class_names))}

    def _list_image_paths(self):
        for entry in sorted(bf.listdir(self.image_directory)):
            full_path = bf.join(self.image_directory, entry)

            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                self.image_paths.append(full_path)
                if bf.basename(full_path).split("_")[0] not in self.class_names:
                    self.class_names.append(bf.basename(full_path).split("_")[0])
            elif bf.isdir(full_path):
                self._list_image_paths(full_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.sorted_classes[bf.basename(path).split("_")[0]]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2

        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.num_classes is not None:
            out_dict["y"] = np.array(label, dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
