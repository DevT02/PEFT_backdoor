import os
import requests
import numpy as np
from PIL import Image
from datasets import load_dataset
import torch
import transformers
import accelerate
from transformers import AutoImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
from peft import PeftConfig, PeftModel


def main():
    prepare_data()


def prepare_data():
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    dataset = load_dataset("food101", split="train[:5000]")
    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch
    
    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
  
def install_dependencies():
    # This function should be run separately if dependencies are not yet installed.
    # !pip install transformers accelerate evaluate datasets git+https://github.com/huggingface/peft -q
    pass

def setup_environment():
    os.environ["HF_HOME"] = "path_to_hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "path_to_transformers_cache"
    transformers_token = "your_transformers_api_token_here"

def load_and_process_dataset(model_checkpoint):
    dataset = load_dataset("food101", split="train[:5000]")
    labels = dataset.features["label"].names
    label2id, id2label = {label: i for i, label in enumerate(labels)}, {i: label for i, label in enumerate(labels)}

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    train_transforms = Compose([
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ])

    def preprocess_images(example_batch):
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    dataset = dataset.map(preprocess_images, batched=True)
    return dataset, label2id, id2label, image_processor

def train_model(dataset, image_processor, label2id):
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    model = PeftModel.from_pretrained("base_model_checkpoint_here", label2id=label2id)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=image_processor
    )
    trainer.train()
    return model

def evaluate_model(model, image_processor, label2id):
    url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(image.convert("RGB"), return_tensors="pt")

    id2label = {i: label for i, label in enumerate(label2id)}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", id2label[predicted_class_idx])

if __name__ == "__main__":
    main()
