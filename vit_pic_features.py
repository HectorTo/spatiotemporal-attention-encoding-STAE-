#get features of each fixation using Vit
import os
import torch
from simple_vit import SimpleViT
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

# Load SimpleViT model
v = SimpleViT(
    image_size=1024,
    patch_size=64,
    num_classes=1000,
    dim=128,
    depth=6,
    heads=16,
    mlp_dim=2048
)

excel_folder = "..."
image_folder = "..."

for root, dirs, files in os.walk(excel_folder):
    all_padded_tensors=[]
    for filename in files:

        if filename.endswith(".xlsx"):
            excel_path = os.path.join(root, filename)

            image_filename = os.path.splitext(filename)[0] + ".jpg"
            image_path = os.path.join(image_folder, image_filename)

            if os.path.exists(image_path):
                print(f"Processing {excel_path} and {image_path}")

                img = Image.open(image_path)

                resize = transforms.Compose([transforms.Resize((1024, 1024))])
                img_resized = resize(img)

                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(img_resized)
                img_tensor.unsqueeze_(0)

                features = v(img_tensor)
                df = pd.read_excel(excel_path, header=None, skiprows=1)
                coordinates_list = [(max(1, min(1023, float(x))), max(1, min(1023, float(y)))) for x, y in
                                    zip(df[1], df[2])]

                patch_size = 64
                patch_coordinates = [(int(x // patch_size), int(y // patch_size)) for x, y in coordinates_list]

                patch_indices = [x + y * (1024 // patch_size) for x, y in patch_coordinates]

                extracted_tokens = [features[0, index, :] for index in patch_indices]

                concatenated_tensor = torch.stack(extracted_tokens, dim=0)

                labels = torch.tensor(df[3].values, dtype=torch.long)

                final_tensor = torch.cat([concatenated_tensor, labels.unsqueeze(1)], dim=1)

                # Ensure final_tensor has size [8, 129]
                if final_tensor.shape[0] > 8:
                    final_tensor = final_tensor[:8, :]
                if final_tensor.shape[1] > 129:
                    final_tensor = final_tensor[:, :129]
                target_size = (8, 129)
                padded_tensor = torch.zeros(target_size)
                padded_tensor[:final_tensor.shape[0], :final_tensor.shape[1]] = final_tensor

                print(padded_tensor.size())

                feature_tensor = padded_tensor[:, :-1]
                label_tensor = padded_tensor[0, -1]

                all_padded_tensors.append(padded_tensor)
                print(f"Feature tensor size: {feature_tensor.size()}")
                print(f"Label tensor size: {label_tensor}")

            else:
                print(f"Corresponding image file not found for {excel_path}")

        final_combined_tensor = torch.stack(all_padded_tensors, dim=0)

        output_folder = "..."
        output_filename = f"{os.path.basename(root)}_output.pt"
        output_path = os.path.join(output_folder, output_filename)
        torch.save(final_combined_tensor, output_path)
