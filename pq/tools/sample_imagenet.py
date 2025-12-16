import os
import random
import shutil
from collections import defaultdict

image_list_file = '/mnt/petrelfs/share/images/meta/train.txt'
image_root_dir = '/mnt/petrelfs/share/images/train/'

save_dir = './sampled_images/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
class_to_images = defaultdict(list)

with open(image_list_file, 'r') as f:
    for line in f:
        relative_path, class_id = line.strip().split()
        full_path = os.path.join(image_root_dir, relative_path)
        class_to_images[class_id].append(full_path)

sampled_images = []
for class_id, image_list in class_to_images.items():
    if len(image_list) >= 2:
        sampled_images.extend(random.sample(image_list, 2))
    else:
        sampled_images.extend(image_list)

for image_path in sampled_images:
    try:
        relative_dir = os.path.dirname(image_path.replace(image_root_dir, '').lstrip('/'))
        target_dir = os.path.join(save_dir, relative_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        file_name = os.path.basename(image_path)
        save_path = os.path.join(target_dir, file_name)
        
        shutil.copy(image_path, save_path)
        print(f"Copied {file_name} to {save_path}")
    except Exception as e:
        print(f"Failed to copy {image_path}: {e}")

print(f"Finished sampling and copying {len(sampled_images)} images.")
