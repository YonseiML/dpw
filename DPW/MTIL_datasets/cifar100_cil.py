import os
import pickle
import numpy as np
import random

from .utils import Datum, DatasetBase, DatasetWrapper
from PIL import Image


class CIFAR100_CIL_Task(DatasetBase):
    """
    CIFAR100 Class-Incremental Learning dataset for a single task.
    Each task contains a subset of classes from CIFAR100.
    """

    dataset_dir = "cifar100"

    def __init__(self, root, task_id, class_order, num_classes_per_task, 
                 num_shots=0, seed=1):
        """
        Args:
            root: dataset root directory
            task_id: current task index (0, 1, 2, ...)
            class_order: list of class indices defining the order of classes
            num_classes_per_task: number of classes per task
            num_shots: number of shots for few-shot learning (-1 for all)
            seed: random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.task_id = task_id
        self.num_classes_per_task = num_classes_per_task

        # Calculate class indices for this task
        start_class = task_id * num_classes_per_task
        end_class = start_class + num_classes_per_task
        self.task_classes = class_order[start_class:end_class]
        
        # Create mapping from original labels to task-local labels (0 to num_classes_per_task-1)
        self.label_mapping = {orig: new for new, orig in enumerate(self.task_classes)}

        # Load train data
        file_path = os.path.join(self.dataset_dir, 'train')
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            trainval_data = entry["data"]
            if "labels" in entry:
                trainval_targets = entry["labels"]
            else:
                trainval_targets = entry["fine_labels"]
        trainval_data = trainval_data.reshape(-1, 3, 32, 32)
        trainval_data = trainval_data.transpose((0, 2, 3, 1))  # convert to HWC

        # Load test data
        file_path = os.path.join(self.dataset_dir, 'test')
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
            test_data = entry["data"]
            if "labels" in entry:
                test_targets = entry["labels"]
            else:
                test_targets = entry["fine_labels"]
        test_data = test_data.reshape(-1, 3, 32, 32)
        test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

        # Load class names
        path = os.path.join(self.dataset_dir, "meta")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            all_classes = data["fine_label_names"]
        all_classes = [s.replace("_", " ") for s in all_classes]

        # Filter and create trainval data for this task's classes
        trainval = []
        for idx in range(trainval_data.shape[0]):
            orig_label = int(trainval_targets[idx])
            if orig_label in self.task_classes:
                new_label = self.label_mapping[orig_label]
                item = Datum(
                    impath=Image.fromarray(trainval_data[idx]),
                    label=new_label,
                    classname=all_classes[orig_label]
                )
                trainval.append(item)

        # Filter and create test data for this task's classes
        test = []
        for idx in range(test_data.shape[0]):
            orig_label = int(test_targets[idx])
            if orig_label in self.task_classes:
                new_label = self.label_mapping[orig_label]
                item = Datum(
                    impath=Image.fromarray(test_data[idx]),
                    label=new_label,
                    classname=all_classes[orig_label]
                )
                test.append(item)

        # Split trainval into train and val (80/20 split)
        train, val = self._split_trainval(trainval, seed)

        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))

        # Set templates
        self.templates = [
            lambda c: f'a photo of a {c}.',
            lambda c: f'a blurry photo of a {c}.',
            lambda c: f'a black and white photo of a {c}.',
            lambda c: f'a low contrast photo of a {c}.',
            lambda c: f'a high contrast photo of a {c}.',
            lambda c: f'a bad photo of a {c}.',
            lambda c: f'a good photo of a {c}.',
            lambda c: f'a photo of a small {c}.',
            lambda c: f'a photo of a big {c}.',
            lambda c: f'a photo of the {c}.',
        ]

        super().__init__(train_x=train, val=val, test=test)

    def _split_trainval(self, trainval, seed, p_val=0.2):
        """Split trainval into train and val sets."""
        random.seed(seed)
        
        # Group by label
        label_to_items = {}
        for item in trainval:
            if item.label not in label_to_items:
                label_to_items[item.label] = []
            label_to_items[item.label].append(item)
        
        train = []
        val = []
        
        for label, items in label_to_items.items():
            random.shuffle(items)
            n_val = int(len(items) * p_val)
            val.extend(items[:n_val])
            train.extend(items[n_val:])
        
        return train, val


def get_cifar100_cil_class_order(seed=1, num_classes=100):
    """
    Get a fixed or random class order for CIFAR100 CIL.
    
    Args:
        seed: random seed for shuffling (use -1 for fixed order)
        num_classes: total number of classes
    
    Returns:
        list of class indices
    """
    class_order = list(range(num_classes))
    if seed >= 0:
        random.seed(seed)
        random.shuffle(class_order)
    return class_order


def get_cifar100_cil_datasets(root, num_tasks, num_classes_per_task, num_shots=-1, seed=1):
    """
    Create all CIFAR100 CIL task datasets.
    
    Args:
        root: dataset root directory
        num_tasks: number of tasks
        num_classes_per_task: number of classes per task
        num_shots: number of shots for few-shot learning
        seed: random seed
    
    Returns:
        list of CIFAR100_CIL_Task objects
    """
    assert num_tasks * num_classes_per_task <= 100, \
        f"num_tasks ({num_tasks}) * num_classes_per_task ({num_classes_per_task}) must be <= 100"
    
    # Get class order (shuffled based on seed)
    class_order = get_cifar100_cil_class_order(seed=seed)
    
    datasets = []
    for task_id in range(num_tasks):
        dataset = CIFAR100_CIL_Task(
            root=root,
            task_id=task_id,
            class_order=class_order,
            num_classes_per_task=num_classes_per_task,
            num_shots=num_shots,
            seed=seed
        )
        datasets.append(dataset)
    
    return datasets
