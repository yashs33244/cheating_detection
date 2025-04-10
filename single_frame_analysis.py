import os
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import argparse

# --- Step 1: Setup paths ---
def setup_folders():
    flagged_folder = 'flagged_frames'
    os.makedirs(flagged_folder, exist_ok=True)
    return flagged_folder

# --- Step 2: Load models ---
def load_models():
    print("Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' or a custom-trained model

    print("Loading CLIP model...")
    # Load the processor and model
    vl_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    vl_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vl_model.to(device)
    
    return yolo_model, vl_processor, vl_model, device

# --- Step 3: Analyze frame ---
def analyze_frame_with_clip(image_path, yolo_results, vl_processor, vl_model, device):
    image = Image.open(image_path).convert("RGB")
    texts = [
        "students looking at each other's papers during exam",
        "students using phones or devices during exam",
        "students passing notes or whispering during exam",
        "students sitting normally and taking exam",
        "students working independently on their exam"
    ]
    
    # Process the image and text with the CLIP processor
    inputs = vl_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the image and text features
    image_features = vl_model.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = vl_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    # Calculate the similarity between image and text features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Get the highest similarity score for cheating behaviors (first 3 prompts)
    cheating_score = similarity[0][:3].max().item()
    normal_score = similarity[0][3:].max().item()
    
    # Check YOLO results for phones and suspicious objects
    num_phones = len([obj for obj in yolo_results[0].boxes.cls.tolist() if int(obj) == 67])  # class 67 is 'cell phone'
    num_people = len([obj for obj in yolo_results[0].boxes.cls.tolist() if int(obj) == 0])   # class 0 is 'person'
    
    # Increase confidence if phones are detected
    if num_phones > 0:
        cheating_score = max(cheating_score, 0.8)
    
    # Use a threshold to determine if cheating is detected
    threshold = 0.4
    if cheating_score > threshold and cheating_score > normal_score:
        behavior_type = texts[similarity[0][:3].argmax().item()]
        confidence = round(cheating_score * 100, 2)
        return True, f"Suspicious activity detected: {behavior_type} (Confidence: {confidence}%, People: {num_people}, Phones: {num_phones})"
    else:
        return False, f"No suspicious activity detected (People: {num_people}, Phones: {num_people})"

# --- Step 4: Generate Summary Report ---
def generate_summary_report(is_suspicious, description):
    print("\n=== Analysis Report ===")
    print(description)
    
    if is_suspicious:
        print("\n=== Suspicious Activity Details ===")
        if "using phones" in description:
            print("Type: Phone usage during exam")
        elif "looking at each other's papers" in description:
            print("Type: Looking at others' papers")
        elif "passing notes" in description:
            print("Type: Passing notes or whispering")
        
        # Extract confidence score
        confidence = float(description.split('Confidence: ')[1].split('%')[0])
        print(f"Confidence: {confidence:.2f}%")
        
        # Extract number of people and phones
        people = int(description.split('People: ')[1].split(',')[0])
        phones = int(description.split('Phones: ')[1].split(')')[0])
        print(f"Number of people detected: {people}")
        print(f"Number of phones detected: {phones}")

# --- Main function ---
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze a single frame for suspicious activity')
    parser.add_argument('image_path', type=str, help='Path to the image file to analyze')
    args = parser.parse_args()
    
    # Setup folders
    flagged_folder = setup_folders()
    
    # Load models
    yolo_model, vl_processor, vl_model, device = load_models()
    
    # Analyze the frame
    print(f"Analyzing frame: {args.image_path}")
    
    # Run YOLO detection
    yolo_results = yolo_model(args.image_path)
    
    # Run CLIP analysis with YOLO results
    is_suspicious, description = analyze_frame_with_clip(args.image_path, yolo_results, vl_processor, vl_model, device)
    print(f"Analysis result: {description}")
    
    # Flag if cheating or suspicious activity is detected
    if is_suspicious:
        frame_name = os.path.basename(args.image_path)
        shutil.copy(args.image_path, os.path.join(flagged_folder, frame_name))
        print(f"Frame saved to {os.path.join(flagged_folder, frame_name)}")
    
    # Generate summary report
    generate_summary_report(is_suspicious, description)

if __name__ == "__main__":
    import shutil  # Import here to avoid circular import
    main() 