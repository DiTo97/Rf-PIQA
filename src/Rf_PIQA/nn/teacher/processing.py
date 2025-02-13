from transformers import AutoImageProcessor


def get_panorama_processor(model_name, image_size):
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = image_size  # e.g., {"height": 256, "width": 1024}
    return processor


def get_constituents_processor(model_name, image_size):
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = image_size  # e.g., {"height": 256, "width": 256}
    return processor
