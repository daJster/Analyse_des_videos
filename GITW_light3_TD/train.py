import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.video_paths = []
        self.bboxes_paths = []
        self.saliency_maps_paths = []
        self.obj_types = []
        self.load_data()

    def load_data(self):
        for icls, cls in enumerate(self.classes):
            cls_path = os.path.join(self.data_dir, cls)
            video_dirs = os.listdir(cls_path)
            for video_dir in video_dirs:
                sub_cls_path = os.path.join(cls_path, video_dir)
                try :
                    video_name = [fname for fname in os.listdir(sub_cls_path) if fname.endswith('.mp4')][0]
                    bbox_name = [fname for fname in os.listdir(sub_cls_path) if fname.endswith('bboxes.txt')][0]
                except IndexError :
                    continue
                
                video_path = os.path.join(sub_cls_path, video_name)
                bbox_path = os.path.join(sub_cls_path, bbox_name)
                saliency_path = os.path.join(sub_cls_path, 'SaliencyMaps')  
                
                self.video_paths.append(video_path)
                self.bboxes_paths.append(bbox_path)
                self.obj_types.append(icls)
                self.saliency_maps_paths.append(saliency_path)
                
                
    def apply_saliency(self, video_path, bbox_path, saliency_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        with open(bbox_path, 'r') as f:
            bboxes_info = [line.split() for line in f.readlines()]

        saliency_applied_frames = []

        for nframe, has_object, *bbox_coords in bboxes_info:
            nframe = int(nframe)
            has_object = int(has_object)

            if has_object:
                bbox_coords = list(map(int, bbox_coords))
                

                # Find the corresponding saliency map using nframe
                saliency_map_files = [map_file for map_file in sorted(os.listdir(saliency_path)) if map_file.endswith("png")]
                saliency_map_file = saliency_map_files[nframe - 1]
                saliency_map_path = os.path.join(saliency_path, saliency_map_file)

                saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
                resized_saliency = cv2.resize(saliency_map, (frames[nframe-1].shape[1], frames[nframe-1].shape[0]))
                saliency_applied = frames[nframe-1] * (resized_saliency[:, :, None] / 255.0)
                
                # apply crop
                cropped_saliency_applied = saliency_applied[bbox_coords[1]:bbox_coords[1] + bbox_coords[3], bbox_coords[0]:bbox_coords[0] + bbox_coords[2]]
                
                if not (saliency_applied.sum(axis=(0, 1, 2)) == 0).all() :
                    saliency_applied_frames.append(torch.tensor(cropped_saliency_applied, dtype=torch.float32) / 255.0)
                
                
        return saliency_applied_frames

    def __len__(self):
        return len(self.video_paths)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224*224, 4)  # Assuming the bounding box has 4 coordinates

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for images, bboxes in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


def get_dataset(data_dir):
    dataset = CustomDataset(data_dir)

    list_all_frames = []
    list_all_types = []
    for idx in range(dataset.__len__()):
        video_path = dataset.video_paths[idx]
        bbox_path = dataset.bboxes_paths[idx]
        obj_type = dataset.obj_types[idx]
        saliency_path = dataset.saliency_maps_paths[idx]

        saliency_applied = dataset.apply_saliency(video_path, bbox_path, saliency_path)

        list_all_frames.extend(save_frames(saliency_applied))
        list_all_types.extend([obj_type] * len(saliency_applied))
        
        
    largest_shape = max(image.shape for image in list_all_frames)
    resized_images = [np.full_like(np.zeros(largest_shape), 255, dtype=np.uint8) for _ in range(len(list_all_frames))]
    for i, image in enumerate(list_all_frames):
        resized_images[i][:image.shape[0], :image.shape[1], :] = image
        
    save_image(torch.tensor(resized_images, dtype=torch.float32), f'output/test-{idx}.jpg')
    # Convert resized_images to PyTorch tensors
    resized_tensors = torch.stack(torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) for image in resized_images)
    classes_tensor = torch.tensor(list_all_types, dtype=torch.long)
    
    # Create a TensorDataset from image tensors and labels
    dataset = TensorDataset(resized_tensors, classes_tensor)
    
    return dataset

def main():
    train_dataset, val_dataset = get_dataset('train'), get_dataset('test')
    print('check')
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load pre-trained VGG-19 model
    vgg19_model = models.vgg19(pretrained=True)
    num_classes = len(set(train_dataset[1]))
    vgg19_model.classifier[6] = nn.Linear(4096, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg19_model.parameters(), lr=0.001, momentum=0.9)


    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg19_model.to(device)

    for epoch in range(num_epochs):
        vgg19_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vgg19_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        vgg19_model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = vgg19_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

def save_frames(frames):
    # os.makedirs(output_path, exist_ok=True)

    # Video writer setup
    # video_path = os.path.join(output_path, f'{unique_name}_video.mp4')
    # height, width, _ = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    list_frames = []
    for frame_idx, frame in enumerate(frames):
        # print(frame_idx, end="\r")
        # Convert torch tensor to NumPy array
        numpy_array = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

        # Transpose the dimensions to make it (3, 222, 162)
        numpy_array = numpy_array.transpose(2, 0, 1)
        list_frames.append(numpy_array)
        # Write the frame to the video
        # video_writer.write(numpy_array)

    # Release the video writer
    # video_writer.release()
    return list_frames
            
            
def merge_videos(input_dir, output_path):
    # Get a list of all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    if not video_files:
        print("No video files found in the directory.")
        return

    # Video writer setup
    first_video_path = os.path.join(input_dir, video_files[0])
    cap = cv2.VideoCapture(first_video_path)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames from each video to the output video
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)

        cap.release()

    # Release the video writer
    video_writer.release()
            
if __name__ == "__main__":
    main()
    # input_directory = 'train_dataset'
    # output_video_path = 'merged_video.mp4'
    # merge_videos(input_directory, output_video_path)
    # ADD CODECARBON
    # data_dir = './train'
    # dataset = CustomDataset(data_dir)
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model = SimpleModel()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train_model(train_loader, model, criterion, optimizer, num_epochs=10)
