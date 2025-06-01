import torch


class Human36MCollator:
    """
    Custom collator for Human3.6M dataset that handles variable-sized images
    by padding them to the largest size in the batch.
    """

    def __call__(self, batch):
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples from the dataset

        Returns:
            Collated batch
        """
        # Calculate max size and pad images and depths (used for augmented data)
        max_height = max(sample["image"].shape[1] for sample in batch)
        max_width = max(sample["image"].shape[2] for sample in batch)

        # Pad images and depths
        padded_images = []
        padded_depths = []

        for sample in batch:
            img = sample["image"]
            depth = sample["depth"]

            # Calculate padding
            pad_h = max_height - img.shape[1]
            pad_w = max_width - img.shape[2]

            if pad_h > 0 or pad_w > 0:
                padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
                padded_depth = torch.nn.functional.pad(depth, (0, pad_w, 0, pad_h))
            else:
                padded_img = img
                padded_depth = depth

            padded_images.append(padded_img)
            padded_depths.append(padded_depth)

        # Create result dictionary
        result = {
            "image": torch.stack(padded_images),
            "depth": torch.stack(padded_depths),
            "keypoints_2d": torch.stack([sample["keypoints_2d"] for sample in batch]),
            "joints_3d": torch.stack([sample["joints_3d"] for sample in batch]),
            "camera_params": [sample["camera_params"] for sample in batch],
            "image_path": [sample["image_path"] for sample in batch],
            "action": [sample["action"] for sample in batch],
            "subaction": [sample["subaction"] for sample in batch],
            "image_size": torch.stack([sample["image_size"] for sample in batch]),
            "frame_idx": [sample["frame_idx"] for sample in batch],
            "padding": [(max_height, max_width)] * len(batch),  # Store padding info
        }

        return result
