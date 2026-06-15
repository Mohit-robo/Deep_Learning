import os
import cv2

def crop_and_save_plates(dataset_dir, split="test", output_dirname="crops"):
    """
    Reads YOLO annotations and crops corresponding plates from the images.
    Saves cropped images to a separate folder.
    """
    split_dir = os.path.join(dataset_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    output_dir = os.path.join(split_dir, output_dirname)
    
    # Ensure directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Split directory structure not found in: {split_dir}")
        print(f"Expected '{images_dir}' and '{labels_dir}'.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading labels from: {labels_dir}")
    print(f"Reading images from: {images_dir}")
    print(f"Crops will be saved to: {output_dir}")
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    
    if not label_files:
        print("No label files (.txt) found.")
        return
        
    total_crops = 0
    total_images_processed = 0
    
    for label_file in sorted(label_files):
        base_name = os.path.splitext(label_file)[0]
        
        # Try finding corresponding image with common extensions
        img_name = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            possible_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(possible_path):
                img_name = base_name + ext
                break
                
        if not img_name:
            print(f"Warning: Image for label file '{label_file}' not found.")
            continue
            
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, label_file)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Read annotations
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        img_crop_count = 0
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # Convert YOLO normalized coords to pixel bounding box coordinates
            # x1, y1, x2, y2 represent top-left and bottom-right corners
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Clip coordinates to image boundaries
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                print(f"Warning: Crop size is 0 for {img_name} at index {idx}")
                continue
                
            # Define output filename
            # E.g. video10_1070_jpg.rf.JLvA6AI1qhyOasZsIjtP_crop_0.jpg
            crop_filename = f"{base_name}_crop_{idx}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            
            # Save crop
            cv2.imwrite(crop_path, crop)
            img_crop_count += 1
            total_crops += 1
            
        if img_crop_count > 0:
            total_images_processed += 1
            
    print("-" * 50)
    print(f"Success! Processed {total_images_processed} images.")
    print(f"Saved {total_crops} cropped plates to '{output_dir}'.")

if __name__ == "__main__":
    dataset_directory = "images/Indian_number_plate"
    crop_and_save_plates(dataset_directory, split="test", output_dirname="crops")
