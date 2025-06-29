import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import warnings
from typing import Dict, Union, Optional

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

class CheXNet(nn.Module):
    """Modified DenseNet121 architecture for chest X-ray classification."""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(CheXNet, self).__init__()
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Freeze early layers if using pretrained weights
        if pretrained:
            for param in self.densenet.parameters():
                param.requires_grad = False
        
        # Modify classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights for new layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class ChestXRayAnalyzer:
    """Chest X-ray analysis system with improved functionality."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            model_path: Path to pretrained weights (optional)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = torch.device(device if device else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        self.classes = ['Normal', 'Pneumonia', 'Pneumothorax', 'Effusion', 'Cardiomegaly']
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Enhanced image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Warm up the model
        self._warmup()

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load model with optional pretrained weights."""
        model = CheXNet(num_classes=len(self.classes), 
                       pretrained=model_path is None)
        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"Loaded weights from {model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}. Using ImageNet-pretrained features.")
        
        return model.to(self.device)

    def _warmup(self):
        """Warm up the model with a dummy input."""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
        except Exception as e:
            print(f"Warmup failed: {e}")

    def analyze_image(self, image_path: str) -> Dict[str, float]:
        """
        Analyze a chest X-ray image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of class probabilities
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
            
            return {cls: float(prob) for cls, prob in zip(self.classes, probs)}
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {cls: 0.0 for cls in self.classes}

    def analyze_pil_image(self, image: Image.Image) -> Dict[str, float]:
        """Analyze a PIL Image directly."""
        try:
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
            return {cls: float(prob) for cls, prob in zip(self.classes, probs)}
        except Exception as e:
            print(f"Error analyzing PIL image: {e}")
            return {cls: 0.0 for cls in self.classes}


if __name__ == "__main__":
    # Example usage with enhanced testing
    analyzer = ChestXRayAnalyzer()
    
    # Test with sample image
    test_image_path = "assets/sample_xray.jpg"
    
    try:
        print("\nAnalyzing image...")
        results = analyzer.analyze_image(test_image_path)
        
        print("\nAnalysis Results:")
        for condition, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{condition:<15}: {prob:.2%}")
            
        # Find most probable condition
        diagnosis = max(results.items(), key=lambda x: x[1])
        print(f"\nMost likely condition: {diagnosis[0]} ({diagnosis[1]:.2%})")
        
    except FileNotFoundError:
        print(f"\nError: Test image not found at {test_image_path}")
        print("Please provide a valid image path or use the analyze_pil_image() method with a PIL Image.")