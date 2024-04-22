from datasets import Dataset, Image
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rust_paths = [
    f"/home/hice1/oupadhyay3/scratch/ml-proj/dataset/Corn/Corn___Common_Rust/image ({i}).JPG"
    for i in range(1, 1193)
]
rust_labels = [0 for _ in range(1192)]

gray_paths = [f"/home/hice1/oupadhyay3/scratch/ml-proj/dataset/Corn/Corn___Gray_Leaf_Spot/image ({i}).JPG" for i in range(1, 514)]
gray_labels = [1 for _ in range(513)]

healthy_paths = [f"/home/hice1/oupadhyay3/scratch/ml-proj/dataset/Corn/Corn___Healthy/image ({i}).JPG" for i in range(1, 1163)]
healthy_labels = [2 for _ in range(1162)]

northern_paths = [f"/home/hice1/oupadhyay3/scratch/ml-proj/dataset/Corn/Corn___Northern_Leaf_Blight/image ({i}).JPG" for i in range(1, 986)]
northern_labels = [3 for _ in range(985)]

# combine all paths and labels
paths = rust_paths + gray_paths + healthy_paths + northern_paths
labels = rust_labels + gray_labels + healthy_labels + northern_labels

for l in labels:
    if l < 0 or l > 3:
        print(l)

assert len(paths) == len(labels)

# dataset
dataset = Dataset.from_dict({
    "image": paths,
    "label": labels
}).cast_column("image", Image())
print(dataset[0]["image"])

splits = dataset.train_test_split(test_size=0.1)
train_dataset = splits["train"]
test_dataset = splits["test"]
splits = train_dataset.train_test_split(test_size=0.1)
train_dataset = splits["train"]
val_dataset = splits["test"]

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# transforms
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose([
    RandomResizedCrop(size),
    ToTensor(),
    normalize,
])
_test_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples["pixel_values"] = [
        _train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

def test_transforms(examples):
    examples["pixel_values"] = [
        _test_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    return examples

train_dataset.set_transform(train_transforms)
val_dataset.set_transform(test_transforms)
test_dataset.set_transform(test_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # print("collate", pixel_values.shape)  # torch.Size([16, 3, 224, 224])
    labels = torch.tensor([example["label"] for example in examples])
    # print("collate", labels.shape)  # torch.Size([16])
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4)
batch = next(iter(train_dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

id2label = {0: "rust", 1: "gray", 2: "healthy", 3: "northern"}
label2id = {v: k for k, v in id2label.items()}

# load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", id2label=id2label, label2id=label2id
)
# untrain all parameters, weights, and biases
# for param in model.parameters():
#     param.requires_grad = False
#     param = torch.nn.Parameter(torch.randn(param.size()))
# for module in model.modules():
#     if hasattr(module, "weight"):
#         module.weight = torch.nn.Parameter(torch.randn(module.weight.size()))
#     if hasattr(module, "bias"):
#         module.bias = torch.nn.Parameter(torch.randn(module.bias.size()))

model.to(device)

args = TrainingArguments(
    "orange-vit-crop-disease-224-pretrained",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="logs",
    remove_unused_columns=False,
)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(y_true=labels, y_pred=predictions))

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train(resume_from_checkpoint=True)

# evaluate
results = trainer.predict(test_dataset)
print(results.metrics)

# save model
trainer.save_model("orange-vit-crop-disease-224")
