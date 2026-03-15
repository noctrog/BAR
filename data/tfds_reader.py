"""ImageNet loading via tensorflow_datasets (tfds).

Alternative to the webdataset path in webdataset_reader.py.
Allows loading ImageNet directly through tfds.load('imagenet2012', ...).
"""

import math
from typing import List

import tensorflow as tf
# Prevent TF from claiming GPU memory — PyTorch owns the GPUs.
tf.config.set_visible_devices([], 'GPU')

import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader, IterableDataset
from PIL import Image

from .webdataset_reader import ImageTransform


class TfdsImageNetIterableDataset(IterableDataset):
    """Wraps tfds imagenet2012 as a PyTorch IterableDataset.

    Handles distributed sharding via tf.distribute.InputContext so each
    (rank, worker) pair gets a unique slice of the data.
    """

    def __init__(
        self,
        tfds_data_dir: str,
        split: str,
        transform,
        shuffle: bool = False,
        shuffle_buffer_size: int = 5000,
    ):
        super().__init__()
        self.tfds_data_dir = tfds_data_dir
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        builder = tfds.builder('imagenet2012', data_dir=tfds_data_dir)
        self.num_examples = builder.info.splits[split].num_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        total_pipelines = world_size * num_workers
        pipeline_id = rank * num_workers + worker_id

        read_config = tfds.ReadConfig(
            input_context=tf.distribute.InputContext(
                num_input_pipelines=total_pipelines,
                input_pipeline_id=pipeline_id,
            ),
        )

        ds = tfds.load(
            'imagenet2012',
            split=self.split,
            data_dir=self.tfds_data_dir,
            read_config=read_config,
            shuffle_files=self.shuffle,
        )

        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer_size)

        for ex in tfds.as_numpy(ds):
            pil_image = Image.fromarray(ex['image']).convert('RGB')
            image_tensor = self.transform(pil_image)
            yield {
                "image": image_tensor,
                "class_id": int(ex['label']),
                "__key__": ex.get('file_name', b'').decode('utf-8') if isinstance(ex.get('file_name', b''), bytes) else str(ex.get('file_name', '')),
            }


class TfdsImageDataset:
    """Mirrors the SimpleImageDataset interface using tfds instead of webdataset.

    Properties: train_dataloader, eval_dataloader, train_dataset, eval_dataset.
    """

    def __init__(
        self,
        tfds_data_dir: str,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.5, 0.5, 0.5],
        normalize_std: List[float] = [0.5, 0.5, 0.5],
    ):
        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std)

        # Train dataset and loader.
        self._train_dataset = TfdsImageNetIterableDataset(
            tfds_data_dir=tfds_data_dir,
            split='train',
            transform=transform.train_transform,
            shuffle=True,
        )
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True if num_workers_per_gpu > 0 else False,
            drop_last=True,
        )

        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * max(num_workers_per_gpu, 1)))
        num_batches = num_worker_batches * max(num_workers_per_gpu, 1)
        num_samples = num_batches * global_batch_size

        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Eval dataset and loader.
        self._eval_dataset = TfdsImageNetIterableDataset(
            tfds_data_dir=tfds_data_dir,
            split='validation',
            transform=transform.eval_transform,
            shuffle=False,
        )
        self._eval_dataloader = DataLoader(
            self._eval_dataset,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=True if num_workers_per_gpu > 0 else False,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader
