import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

# 1. Load Dataset
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split = 'train')

# 2. Define Image Size and Preprocessing
image_size = 64
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

# 3. Set Transform for Dataset
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
dataset.set_transform(transform)

# 4. Create DataLoader
batch_size = 10000
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)


# 6. Function to Display Images
def show_images(images):
    plt.figure(figsize=(12, 6))
    plt.imshow(make_grid(images * 0.5 + 0.5).permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# 7. Get and Display a Batch of Images
batch = next(iter(train_dataloader))
images_to_save = batch['images'][:]

# 9. Save Images
save_dir = "images"  # Create a directory to save images
os.makedirs(save_dir, exist_ok=True)

for i, image in enumerate(images_to_save):
    # Un-normalize and convert to PIL Image
    image = (image * 0.5 + 0.5).clamp(0, 1)  # Clamp to ensure valid pixel values
    image = transforms.ToPILImage()(image.cpu())  # Move to CPU and convert to PIL Image

    # Save as JPEG
    save_path = os.path.join(save_dir, f"butterfly_{i}.jpg")
    image.save(save_path)