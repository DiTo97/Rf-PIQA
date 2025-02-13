import torch


def collate(batch, panorama_processor, constituents_processor):
    """
    Given a batch (list of dicts with keys 'high_res_image' and 'low_res_images'),
    process and pad the low-res images and stack the high-res images.
    """
    panoramas = [item['panorama'] for item in batch]
    constituents_batch = [item['constituents'] for item in batch]

    # Determine the maximum number of low-res images in the batch
    max_num_low_res = max(len(imgs) for imgs in constituents_batch)
    padded_constituents_batch = []
    for imgs_batch in constituents_batch:
        # Process the list of images into a tensor of shape (num_images, C, H, W)
        processed = constituents_processor(imgs_batch, return_tensors="pt").pixel_values
        if processed.shape[0] < max_num_low_res:
            pad_size = max_num_low_res - processed.shape[0]
            pad_tensor = torch.zeros((pad_size, *processed.shape[1:]))
            processed = torch.cat([processed, pad_tensor], dim=0)
        padded_constituents_batch.append(processed)
    
    # Create padding masks (1 for valid images, 0 for padded entries)
    padding_mask = torch.tensor([
        [1] * len(imgs) + [0] * (max_num_low_res - len(imgs))
        for imgs in constituents_batch
    ])
    
    # Process high-res images (the panorama)
    panoramas = panorama_processor(panoramas, return_tensors="pt").pixel_values
    constituents_batch = torch.stack(padded_constituents_batch)
    return panoramas, constituents_batch, padding_mask
