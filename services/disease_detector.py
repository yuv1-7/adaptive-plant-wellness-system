import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from typing import Dict, List, Tuple
import os
import json
from pathlib import Path


class PlantDiseaseDetector:
    """Service for detecting plant diseases using fine-tuned ResNet"""
    
    def __init__(self, model_path: str = "models/disease_model_best.pth", 
                 class_mapping_path: str = "models/class_mapping.json"):
        """
        Initialize the disease detector
        
        Args:
            model_path: Path to the trained model weights
            class_mapping_path: Path to class mapping JSON file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load class mapping
        self.class_to_idx, self.idx_to_class = self._load_class_mapping(class_mapping_path)
        self.num_classes = len(self.class_to_idx)
        
        # Load model
        self.model = self._load_model()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_class_mapping(self, class_mapping_path: str) -> Tuple[Dict, Dict]:
        """Load class mapping from JSON file"""
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return class_to_idx, idx_to_class
        else:
            # Default fallback - will be updated when model is trained
            print(f"Warning: Class mapping not found at {class_mapping_path}")
            print("Model will need to be trained first.")
            return {}, {}
    
    def _load_model(self) -> nn.Module:
        """Load and configure the ResNet model"""
        # Create model architecture
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        # Load trained weights if available
        if os.path.exists(self.model_path):
            try:
                model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                model.eval()
                print(f"✓ Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        else:
            print(f"Warning: Model not found at {self.model_path}")
            print("Please train the model first using train_universal.py")
        
        model = model.to(self.device)
        return model
    
    async def detect_disease(self, image_data: bytes) -> Dict:
        """
        Detect plant disease from image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with detection results
        """
        try:
            if not self.class_to_idx:
                return {
                    'success': False,
                    'error': 'Model not trained. Please train the model first.',
                    'predictions': []
                }
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(3, self.num_classes))
            
            results = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                class_name = self.idx_to_class[idx.item()]
                plant_name, disease = self._parse_class_name(class_name)
                
                results.append({
                    'plant': plant_name,
                    'disease': disease,
                    'confidence': float(prob.item() * 100),
                    'is_healthy': 'healthy' in disease.lower()
                })
            
            return {
                'success': True,
                'predictions': results,
                'top_prediction': results[0]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': []
            }
    
    def _parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """
        Parse class name into plant and disease components
        Handles multiple formats:
        - Plant___Disease (PlantVillage format)
        - Disease (PlantDoc format)
        - Plant_Disease (other formats)
        """
        # PlantVillage format: Apple___Apple_scab
        if '___' in class_name:
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ')
            return plant, disease
        
        # Check for known plant prefixes
        plant_prefixes = [
            'Apple', 'Tomato', 'Grape', 'Corn', 'Potato', 'Pepper',
            'Peach', 'Cherry', 'Strawberry', 'Orange', 'Soybean',
            'Raspberry', 'Blueberry', 'Squash', 'Rose', 'Chilli',
            'Tulsi', 'Basil', 'Mint'
        ]
        
        for prefix in plant_prefixes:
            if class_name.startswith(prefix):
                # Extract plant name
                plant = prefix
                # Rest is disease
                disease = class_name[len(prefix):].strip('_').replace('_', ' ')
                if not disease:
                    disease = 'Unknown condition'
                return plant, disease
        
        # Default: treat entire name as disease
        return 'Unknown plant', class_name.replace('_', ' ')
    
    def get_supported_classes(self) -> List[str]:
        """Get list of all supported disease classes"""
        return list(self.class_to_idx.keys())
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'num_classes': self.num_classes,
            'model_loaded': os.path.exists(self.model_path),
            'device': str(self.device),
            'supported_classes': self.get_supported_classes()
        }