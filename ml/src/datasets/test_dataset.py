from datasets.dataset import ChestXrayDataset
from datasets.transforms import train_transform

dataset = ChestXrayDataset(
    csv_file="../data/train.csv",
    transform=train_transform
)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print(image.shape)
print(label.shape)
print(label)