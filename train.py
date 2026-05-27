"""
Complete training pipeline for road hazard detection model (Potholes + Speed Breakers)
FIX: This version bypasses the downloaded data.yaml file and creates its own,
     to fix the 'Missing names or nc keys' error.
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path

class RoadHazardTrainer:
    def __init__(self, dataset_path='./road_hazard_dataset'):
        self.dataset_path = Path(dataset_path)
        self.model = None
        
    def setup_dataset_config(self, class_names=['Pothole', 'SpeedBreaker']):
        """
        Create data.yaml configuration file
        
        Args:
            class_names: List of pothole types to detect
        """
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        # This will create a new 'data.yaml' or overwrite the broken one
        config_path = self.dataset_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ New dataset configuration saved to {config_path}")
        return config_path
    
    def download_sample_dataset(self):
        """
        Provides instructions if the dataset is missing.
        """
        print("=" * 60)
        print("DATASET NOT FOUND!")
        print("=" * 60)
        print("\nPlease run the 'download_dataset.py' script first to get the")
        print("correct dataset with Potholes and Speed Breakers.")
        print("\n 1. Get your API key from Roboflow (https://app.roboflow.com/settings/api)")
        print(" 2. Put the key in 'download_dataset.py'")
        print(" 3. Run: python download_dataset.py")
        print(" 4. Once downloaded, run this training script again.")
        print("=" * 60)
    
    def validate_dataset(self):
        """
        Check if dataset structure is correct.
        <--- MODIFIED: This function no longer reads the broken data.yaml.
        It only checks if the image and label folders exist.
        """
        required_paths = [
            self.dataset_path / 'train' / 'images',
            self.dataset_path / 'train' / 'labels',
            self.dataset_path / 'valid' / 'images',
            self.dataset_path / 'valid' / 'labels',
        ]
        
        for path in required_paths:
            if not path.exists():
                print(f"✗ Missing: {path}")
                return False  # Return boolean False
            
            files = list(path.glob('*'))
            if len(files) == 0:
                print(f"✗ Empty directory: {path}")
                return False  # Return boolean False
            
            print(f"✓ Found {len(files)} files in {path.name}")
        
        print("✓ All required image/label folders found.")
        return True  # Return boolean True
    
    def train_model(self, 
                    class_names, # We will pass this in from main
                    model_size='m',  # n, s, m, l, x
                    epochs=55,
                    img_size=640,
                    batch_size=16,
                    device='0'):
        """
        Train pothole detection model
        """
        
        # Setup configuration
        # <--- THIS IS THE FIX ---
        # This call will create a new, valid data.yaml file
        # and overwrite the broken one.
        config_path = self.setup_dataset_config(class_names=class_names)
        
        # Load pre-trained model
        print(f"\n📦 Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # Train
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"   Image size: {img_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")
        
        results = self.model.train(
            data=str(config_path), # Use our newly created config file
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project='road_hazard_training1',
            name=f'hazard_yolov8{model_size}',
            patience=20,
            save=True,
            plots=True,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
            perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, warmup_momentum=0.8, box=7.5, cls=0.5, dfl=1.5,
        )
        
        print("\n✓ Training completed!")
        return results
    
    def validate_model(self):
        """Validate trained model"""
        if self.model is None:
            print("❌ No model loaded. Train a model first.")
            return None
        
        print("\n📊 Validating model...")
        metrics = self.model.val()
        
        print("\n=== Validation Metrics ===")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, formats=['onnx', 'torchscript']):
        """
        Export model to different formats for deployment
        """
        if self.model is None:
            print("❌ No model loaded.")
            return
        
        print("\n📤 Exporting model...")
        for fmt in formats:
            print(f"   Exporting to {fmt}...")
            self.model.export(format=fmt)
            print(f"   ✓ {fmt} export complete")
    
    def test_inference(self, image_path):
        """
        Test model on a single image
        """
        if self.model is None:
            print("❌ No model loaded.")
            return
        
        print(f"\n🔬 Testing inference on {image_path}...")
        results = self.model(image_path)
        
        for r in results:
            print(f"\nDetected {len(r.boxes)} objects:")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id] # Get class name
                conf = float(box.conf[0])
                print(f"   - Class: {cls_name}, Confidence: {conf:.2f}")
        
        results[0].save(filename='test_result.jpg')
        print("✓ Result saved to test_result.jpg")


def main():
    """Main training workflow"""
    print("=" * 60)
    print("    ROAD HAZARD DETECTION MODEL TRAINING PIPELINE")
    print("         (Potholes + Speed Breakers)")
    print("=" * 60)
    
    # <--- MODIFIED: Make sure this path is correct!
    # This should be the folder that contains the 'train' and 'valid' folders.
    dataset_dir = './Pothole-and-Speed-Breaker-Dataset-400B-2' 
    
    trainer = RoadHazardTrainer(dataset_path=dataset_dir)
    
    # Check dataset
    print("\n🔍 Validating dataset structure...")
    # <--- MODIFIED: 'is_valid' is now a boolean (True/False)
    is_valid = trainer.validate_dataset()
    
    if not is_valid:
        print("\n❌ Dataset validation failed!")
        trainer.download_sample_dataset()
        return
    
    # <--- MODIFIED: We now manually define the class names
    # This MUST match the order from the Roboflow dataset
    # 0 = Pothole
    # 1 = SpeedBreaker
    class_names = ['Pothole', 'SpeedBreaker']
    
    # Configuration
    print("\n📝 Training Configuration:")
    print("-" * 60)
    
    # 's' (small) is a good balance of speed and accuracy.
    # 'n' (nano) is fastest but less accurate.
    MODEL_SIZE = 'm'
    EPOCHS = 55      # Start with 50, increase to 100+ for better results
    IMG_SIZE = 640
    BATCH_SIZE = 16   # Reduce to 8 or 4 if you get "Out of Memory" errors
    DEVICE = '0'      # Use '0' for GPU, 'cpu' for CPU
    
    print(f"Model Size: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {class_names} (Manually set)") # <--- MODIFIED
    print("-" * 60)
    
    # Train model
    results = trainer.train_model(
        class_names=class_names, # <--- Pass in our manual list
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    if results is None:
        return
    
    # Validate model
    trainer.validate_model()
    
    # (Optional) Export and Test
    trainer.export_model(formats=['onnx'])
    
    # Find a test image to use
    test_image_path = list((Path(dataset_dir) / 'valid' / 'images').glob('*.jpg'))[0]
    if test_image_path:
        trainer.test_inference(image_path=str(test_image_path))
    
    print("\n" + "=" * 60)
    print("✓ TRAINING PIPELINE COMPLETED!")
    print("=" * 60)
    print("\nYour trained model is saved in:")
    print(f"   road_hazard_training/hazard_yolov8{MODEL_SIZE}/weights/best.pt")
    print("\nUpdate your live detection script to use this new model path!")
    print("=" * 60)


if __name__ == "__main__":
    main()