import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import logging
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from skimage import feature
from sklearn.metrics import f1_score

# -----------------------------
# Set up Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Edge Enhancement Class
# -----------------------------
class EdgeEnhancement:
    def __init__(self, 
                 sigma=2.0,
                 low_threshold=0.1,
                 high_threshold=0.3,
                 enhancement_strength=0.5,
                 use_multiscale=True):
        """
        Enhanced edge detection and enhancement class.
        
        Args:
            sigma (float): Gaussian smoothing parameter
            low_threshold (float): Lower threshold for Canny edge detection
            high_threshold (float): Higher threshold for Canny edge detection
            enhancement_strength (float): Strength of edge enhancement (0-1)
            use_multiscale (bool): Whether to use multiscale edge detection
        """
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.enhancement_strength = enhancement_strength
        self.use_multiscale = use_multiscale
        
    def apply_multiscale_edge_detection(self, image_np):
        """Apply edge detection at multiple scales and combine results."""
        scales = [0.5, 1.0, 2.0]
        edge_maps = []
        
        for scale in scales:
            sigma = self.sigma * scale
            edges = [
                feature.canny(
                    image_np[..., i],
                    sigma=sigma,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold
                ) for i in range(3)
            ]
            edge_maps.append(np.stack(edges, axis=-1))
            
        # Combine edge maps using maximum response
        combined_edges = np.maximum.reduce(edge_maps)
        return combined_edges
    
    def apply_selective_enhancement(self, image_np, edge_map):
        """Enhance edges while preserving important image details."""
        # Calculate image structure tensor
        gx = np.gradient(image_np, axis=1)
        gy = np.gradient(image_np, axis=0)
        
        # Calculate local structure strength
        structure_strength = np.sqrt(gx**2 + gy**2)
        
        # Create adaptive enhancement mask
        enhancement_mask = structure_strength * edge_map
        
        # Normalize enhancement mask
        enhancement_mask = enhancement_mask / (enhancement_mask.max() + 1e-8)
        
        # Apply selective enhancement
        enhanced_image = image_np + (self.enhancement_strength * enhancement_mask * edge_map)
        return np.clip(enhanced_image, 0.0, 1.0)
        
    def __call__(self, image):
        """
        Apply edge enhancement to an image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Edge-enhanced image
        """
        # Convert PIL Image to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Apply edge detection
        if self.use_multiscale:
            edge_map = self.apply_multiscale_edge_detection(image_np)
        else:
            edges = [
                feature.canny(
                    image_np[..., i],
                    sigma=self.sigma,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold
                ) for i in range(3)
            ]
            edge_map = np.stack(edges, axis=-1)
        
        # Apply selective enhancement
        enhanced_image = self.apply_selective_enhancement(image_np, edge_map)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray((enhanced_image * 255).astype(np.uint8))
        return enhanced_image

# -----------------------------
# Positional Encoding Class
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# -----------------------------
# Unified Multi-Modal Transformer
# -----------------------------
class UnifiedMultiModalTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, seq_len, feature_dim]
        Returns:
            [batch_size, seq_len, feature_dim]
        """
        # Transpose for transformer [seq_len, batch_size, feature_dim]
        embeddings = embeddings.transpose(0, 1)
        transformed = self.transformer_encoder(embeddings)
        return self.norm(transformed.transpose(0, 1))  # [batch_size, seq_len, feature_dim]

# -----------------------------
# Weighted Cross-Entropy Loss
# -----------------------------
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, attribute_weights=None):
        super().__init__()
        self.attribute_weights = attribute_weights if attribute_weights else {}
    
    def forward(self, outputs, targets):
        loss = 0.0
        losses_dict = {}
        
        for i, (attr_name, output) in enumerate(outputs.items()):
            # Get weight for this attribute (default to 1.0 if not specified)
            attr_weight = self.attribute_weights.get(attr_name, 1.0)
            target = targets[:, i]
            
            # Debugging checks
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error(f"NaN or Inf in outputs for attribute {attr_name}")
            if target.min() < -1 or target.max() >= output.size(1):
                logger.error(f"Invalid target values for attribute {attr_name}: min {target.min()}, max {target.max()}")
            
            # Calculate weighted cross-entropy loss for this attribute
            attr_loss = F.cross_entropy(output, target, ignore_index=-1)
            weighted_loss = attr_weight * attr_loss
            
            loss += weighted_loss
            losses_dict[attr_name] = weighted_loss.item()
        
        return loss, losses_dict

# -----------------------------
# Weight Adjustment Function
# -----------------------------
def adjust_weights(val_metrics, current_weights, adjustment_factor=0.1):
    """
    Adjust attribute weights based on validation F1 scores
    """
    new_weights = {}
    # Calculate mean F1 score across all attributes
    f1_scores = [score for attr, score in val_metrics.items()]
    mean_f1 = np.mean(f1_scores)
    
    for attr, f1 in val_metrics.items():
        current_weight = current_weights.get(attr, 1.0)
        if f1 < mean_f1:
            # Increase weight for underperforming attributes
            new_weights[attr] = current_weight * (1 + adjustment_factor)
        else:
            # Slightly decrease weight for well-performing attributes
            new_weights[attr] = current_weight * (1 - adjustment_factor * 0.5)
    
    # Normalize weights to prevent explosion
    weight_sum = sum(new_weights.values())
    new_weights = {k: v / weight_sum * len(new_weights) for k, v in new_weights.items()}
    
    return new_weights

# -----------------------------
# Custom Women T-Shirt Dataset with Edge Enhancement
# -----------------------------
class CustomWomenTShirtDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, combined_embeddings=None, 
                 attribute_names=None, edge_enhancement=False, edge_enhancer_params=None):
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform
        self.edge_enhancement = edge_enhancement

        # Use the passed attribute names
        self.attribute_names = attribute_names
        self.attr_cols = [f'attr_{i+1}' for i in range(len(self.attribute_names))]  # ['attr_1', 'attr_2', ...]

        required_columns = ['id', 'image_path'] + self.attr_cols + ['len']
        missing_columns = [col for col in required_columns if col not in self.annotations.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing in {csv_path}: {missing_columns}")

        self.labels = self.annotations[self.attr_cols].values
        self.image_paths = self.annotations['image_path'].values.tolist()

        # Create attribute value to index mapping for labels
        # Since we have combined embeddings, extract mapping from combined_embeddings
        self.attribute_value_to_idx = {}
        for attr in self.attribute_names:
            self.attribute_value_to_idx[attr] = {}
            for idx, entry in enumerate(combined_embeddings.get(attr, [])):
                value = entry["value"]
                self.attribute_value_to_idx[attr][value] = idx  # Assuming the order matches

        # Initialize edge enhancer if enabled
        if self.edge_enhancement:
            if edge_enhancer_params is None:
                edge_enhancer_params = {}
            self.edge_enhancer = EdgeEnhancement(**edge_enhancer_params)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply edge enhancement if enabled
        if self.edge_enhancement:
            edge_enhanced_image = self.edge_enhancer(image)
        else:
            edge_enhanced_image = image

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            edge_enhanced_image = self.transform(edge_enhanced_image)

        # Convert attribute values to indices
        label_indices = []
        for i, attr_col in enumerate(self.attr_cols):
            attr_name = self.attribute_names[i]
            attr_value = str(self.annotations.iloc[idx][attr_col])
            value_to_idx = self.attribute_value_to_idx.get(attr_name, {})
            if not value_to_idx:
                # Skip this attribute if the mapping is empty
                label_idx = -1
            else:
                label_idx = value_to_idx.get(attr_value, -1)  # Use -1 for unknown values
            label_indices.append(label_idx)
        labels = torch.tensor(label_indices, dtype=torch.long)

        return image, edge_enhanced_image, labels

# -----------------------------
# Residual FFNN Class
# -----------------------------
class ResidualFFN(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),          # Activation Function
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.ffn(x)  # Residual Connection

# -----------------------------
# Enhanced Attribute Module with Unified Multi-Modal Transformer
# -----------------------------
class EnhancedAttributeModule(nn.Module):
    def __init__(self, feature_dim, roberta_dim, sentence_dim, num_heads=8, dropout=0.2):
        super().__init__()
        
        # Projection layers for different embeddings
        self.roberta_proj = nn.Linear(roberta_dim, feature_dim)
        self.sentence_proj = nn.Linear(sentence_dim, feature_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(feature_dim, max_len=3)
        
        # Unified Multi-Modal Transformer
        self.multi_modal_transformer = UnifiedMultiModalTransformer(feature_dim, num_heads, num_layers=2, dropout=dropout)
        
        # Feature fusion layer with GELU
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),          # Changed from ReLU to GELU
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.output_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, visual_features, roberta_emb, sentence_emb):
        batch_size = visual_features.size(0)
        
        # Project embeddings to common space
        roberta_proj = self.roberta_proj(roberta_emb)        # [batch_size, feature_dim]
        sentence_proj = self.sentence_proj(sentence_emb)      # [batch_size, feature_dim]
        
        # Stack embeddings for unified transformer [batch_size, seq_len=3, feature_dim]
        embeddings = torch.stack([visual_features, roberta_proj, sentence_proj], dim=1)
        
        # Apply positional encoding
        embeddings = self.pos_encoder(embeddings)
        
        # Apply unified transformer
        transformed = self.multi_modal_transformer(embeddings)  # [batch_size, seq_len=3, feature_dim]
        
        # Aggregate transformed embeddings (e.g., mean)
        aggregated = transformed.mean(dim=1)  # [batch_size, feature_dim]
        
        # Fuse features
        fused = self.fusion(torch.cat([visual_features, roberta_proj, sentence_proj], dim=-1))  # [batch_size, feature_dim]
        
        # Final normalization
        output = self.output_norm(fused + aggregated)  # [batch_size, feature_dim]
        
        return output

# -----------------------------
# Enhanced Classification Head
# -----------------------------
class EnhancedClassificationHead(nn.Module):
    def __init__(self, feature_dim, roberta_dim, sentence_dim, num_classes, dropout=0.2):
        super().__init__()
        
        self.attribute_module = EnhancedAttributeModule(
            feature_dim=feature_dim,
            roberta_dim=roberta_dim,
            sentence_dim=sentence_dim,
            dropout=dropout
        )
        
        # Enhanced classifier with residual connections using GELU
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),          # GELU activation
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),          # GELU activation
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, features, roberta_emb, sentence_emb):
        # Get enhanced features through cross-attention
        enhanced = self.attribute_module(features, roberta_emb, sentence_emb)  # [batch_size, feature_dim]
        
        # Apply classification
        output = self.classifier(enhanced)  # [batch_size, num_classes]
        
        return output

# -----------------------------
# ResNet Feature Extractor
# -----------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial conv layer
        self.input_conv = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create residual blocks with skip connections
        self.layer1 = self._make_layer(channels[0], channels[0], 2)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2)
        
        self.num_features = channels[3]
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block handles dimension change
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ))
        
        # Skip connection for dimension matching
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        # Wrap the first block with its shortcut
        layers[0] = nn.ModuleDict({
            'block': layers[0],
            'shortcut': shortcut
        })
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ))
            # No dimension change in additional blocks
            layers[-1] = nn.ModuleDict({
                'block': layers[-1],
                'shortcut': nn.Identity()
            })
                
        return nn.ModuleList(layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.maxpool(self.relu(self.bn1(self.input_conv(x))))
        
        # Store intermediate features for skip connections
        features = []
        
        # Process through layers with skip connections
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block_dict in layer:
                identity = x
                x = block_dict['block'](x)
                identity = block_dict['shortcut'](identity)
                x = self.relu(x + identity)
                features.append(x)
        
        return x, features

# -----------------------------
# Combined Model with Enhanced Classification Heads
# -----------------------------
class EnhancedCombinedModel(nn.Module):
    def __init__(
        self,
        image_model1: nn.Module,
        image_model2: nn.Module,
        combined_embeddings: dict,
        dropout_prob: float = 0.4,
        mlp_dim: int = 512,
        normalization: str = 'layernorm'
    ):
        super(EnhancedCombinedModel, self).__init__()
        self.image_model1 = image_model1
        self.image_model2 = image_model2

        # Add the custom ResNet-style feature extractor
        self.feature_extractor = ResNetFeatureExtractor()
        
        # Feature dimensions
        image_feature_dim1 = self.image_model1.num_features
        image_feature_dim2 = self.image_model2.num_features
        custom_feature_dim = self.feature_extractor.num_features

        # Projection layers for each model to a common dimension
        self.projection1 = nn.Linear(image_feature_dim1, mlp_dim)
        self.projection2 = nn.Linear(image_feature_dim2, mlp_dim)
        self.projection3 = nn.Linear(custom_feature_dim, mlp_dim)

        # Feature fusion attention
        self.feature_fusion = nn.MultiheadAttention(mlp_dim, num_heads=8, dropout=dropout_prob)

        # Initialize attribute_names with valid embeddings
        self.attribute_names = []
        self.roberta_embeddings = nn.ModuleDict()
        self.sentence_embeddings = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()

        for attr, values in combined_embeddings.items():
            if not values:
                logger.warning(f"No embedding values found for attribute {attr}. Skipping.")
                continue
            self.attribute_names.append(attr)

            first_entry = values[0]
            roberta_dim = len(first_entry['roberta_embedding'])
            sentence_dim = len(first_entry['sentence_transformer_embedding'])
            num_classes = len(values)
            
            # Prepare embedding weights from JSON structure
            roberta_weight = torch.tensor([entry['roberta_embedding'] for entry in values], dtype=torch.float32)  # [num_classes, roberta_dim]
            sentence_weight = torch.tensor([entry['sentence_transformer_embedding'] for entry in values], dtype=torch.float32)  # [num_classes, sentence_dim]
            
            self.roberta_embeddings[attr] = nn.Embedding.from_pretrained(roberta_weight, freeze=False)
            self.sentence_embeddings[attr] = nn.Embedding.from_pretrained(sentence_weight, freeze=False)
            
            # Create classification heads for each attribute
            self.classification_heads[attr] = EnhancedClassificationHead(
                feature_dim=mlp_dim,
                roberta_dim=roberta_dim,
                sentence_dim=sentence_dim,
                num_classes=num_classes,
                dropout=dropout_prob
            )

        # MLP Block for feature refinement with residual connections
        if normalization.lower() == 'layernorm':
            norm_layer1 = nn.LayerNorm(mlp_dim)
            norm_layer2 = nn.LayerNorm(mlp_dim)
        elif normalization.lower() == 'batchnorm':
            norm_layer1 = nn.BatchNorm1d(mlp_dim)
            norm_layer2 = nn.BatchNorm1d(mlp_dim)
        else:
            raise ValueError("Normalization must be either 'layernorm' or 'batchnorm'")

        self.mlp_block = nn.Sequential(
            nn.Linear(mlp_dim * 4, mlp_dim),
            nn.ReLU(),
            norm_layer1,
            ResidualFFN(mlp_dim, dropout_prob),
            nn.ReLU(),
            norm_layer2,
            nn.Dropout(dropout_prob)
        )
        
        # Corrected total number of skip feature channels
        sum_channels = 2*64 + 2*128 + 2*256 + 2*512  # = 1920
        self.skip_projection = nn.Linear(sum_channels, mlp_dim)

    def forward(self, image1, image2):
        # Extract features from both ConvNeXt models
        features1 = self.image_model1.forward_features(image1)  # [batch_size, feature_dim, H, W]
        features1 = features1.mean(dim=[2, 3])  # Global Average Pooling -> [batch_size, feature_dim]
        
        features2 = self.image_model2.forward_features(image2)  # [batch_size, feature_dim, H, W]
        features2 = features2.mean(dim=[2, 3])  # Global Average Pooling -> [batch_size, feature_dim]
        
        # Extract features from custom ResNet feature extractor
        features3, skip_features = self.feature_extractor(image1)
        features3 = features3.mean(dim=[2, 3])  # [batch_size, custom_feature_dim]
        
        # Project all features to common dimension
        proj1 = self.projection1(features1)  # [batch_size, mlp_dim]
        proj2 = self.projection2(features2)  # [batch_size, mlp_dim]
        proj3 = self.projection3(features3)  # [batch_size, mlp_dim]
        
        # Stack features for attention-based fusion
        # Shape: [seq_len=3, batch_size, mlp_dim]
        stacked_features = torch.stack([proj1, proj2, proj3], dim=0)  # [3, batch_size, mlp_dim]
        fused_features, _ = self.feature_fusion(stacked_features, stacked_features, stacked_features)  # [3, batch_size, mlp_dim]
        
        # Combine fused features by averaging across the sequence length
        fused_features = fused_features.mean(dim=0)  # [batch_size, mlp_dim]
        
        # Incorporate skip connection features from ResNet
        # Aggregate skip features: list of [batch_size, channels, H, W]
        # Perform Global Average Pooling and concatenate them
        skip_feature = torch.cat([f.mean(dim=[2, 3]) for f in skip_features], dim=1)  # [batch_size, sum_channels=1920]
        
        # Project skip features to mlp_dim (using the pre-defined skip_projection)
        skip_features_projected = self.skip_projection(skip_feature)  # [batch_size, mlp_dim]
        
        # Final feature combination with skip connections
        combined_features = fused_features + skip_features_projected  # [batch_size, mlp_dim]
        
        # Pass through MLP block
        refined_features = self.mlp_block(torch.cat([proj1, proj2, proj3, combined_features], dim=1))  # [batch_size, mlp_dim]
        
        outputs = {}
        for attr in self.attribute_names:
            # Get the classification head for this attribute
            classification_head = self.classification_heads[attr]
            
            # Get the attribute embeddings
            roberta_emb = self.roberta_embeddings[attr].weight  # [num_classes, roberta_dim]
            sentence_emb = self.sentence_embeddings[attr].weight  # [num_classes, sentence_dim]
            
            # Calculate average attribute embeddings
            avg_roberta_emb = roberta_emb.mean(dim=0)  # [roberta_dim]
            avg_sentence_emb = sentence_emb.mean(dim=0)  # [sentence_dim]
            
            # Expand to match batch size
            batch_size = refined_features.size(0)
            avg_roberta_emb = avg_roberta_emb.unsqueeze(0).expand(batch_size, -1)  # [batch_size, roberta_dim]
            avg_sentence_emb = avg_sentence_emb.unsqueeze(0).expand(batch_size, -1)  # [batch_size, sentence_dim]
            
            # Generate predictions using the classification head
            logits = classification_head(refined_features, avg_roberta_emb, avg_sentence_emb)  # [batch_size, num_classes]
            
            outputs[attr] = logits

        return outputs

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def calculate_metrics(outputs, targets, attribute_names):
    """
    Calculate F1 scores for each attribute
    """
    metrics = {}
    for i, attr in enumerate(attribute_names):
        output = outputs[attr]
        target = targets[:, i]
        
        # Get predictions
        _, predicted = torch.max(output, 1)
        
        # Calculate F1 score (ignore invalid indices)
        mask = target != -1
        if mask.sum() > 0:
            f1 = f1_score(
                target[mask].cpu().numpy(),
                predicted[mask].cpu().numpy(),
                average='weighted',
                zero_division=0
            )
            metrics[attr] = f1
        else:
            metrics[attr] = 0.0
            
    return metrics

def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_train_preds = {attr: [] for attr in model.attribute_names}
    all_train_targets = {attr: [] for attr in model.attribute_names}

    for images, edge_images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        edge_images = edge_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images, edge_images)
            loss, _ = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Collect predictions for metrics
        for i, attr in enumerate(model.attribute_names):
            output = outputs[attr]
            target = labels[:, i]
            _, predicted = torch.max(output, 1)
            mask = target != -1
            all_train_preds[attr].extend(predicted[mask].cpu().numpy())
            all_train_targets[attr].extend(target[mask].cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    train_metrics = {}
    for attr in model.attribute_names:
        if len(all_train_targets[attr]) > 0:
            f1 = f1_score(all_train_targets[attr], all_train_preds[attr], average='weighted', zero_division=0)
            train_metrics[attr] = f1

    return epoch_loss, train_metrics

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    val_loss = 0.0
    all_val_preds = {attr: [] for attr in model.attribute_names}
    all_val_targets = {attr: [] for attr in model.attribute_names}

    with torch.no_grad():
        for images, edge_images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            edge_images = edge_images.to(device)
            labels = labels.to(device)

            outputs = model(images, edge_images)
            loss, _ = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect predictions for metrics
            for i, attr in enumerate(model.attribute_names):
                output = outputs[attr]
                target = labels[:, i]
                _, predicted = torch.max(output, 1)
                mask = target != -1
                all_val_preds[attr].extend(predicted[mask].cpu().numpy())
                all_val_targets[attr].extend(target[mask].cpu().numpy())

    avg_loss = val_loss / len(dataloader)
    val_metrics = {}
    for attr in model.attribute_names:
        if len(all_val_targets[attr]) > 0:
            f1 = f1_score(all_val_targets[attr], all_val_preds[attr], average='weighted', zero_division=0)
            val_metrics[attr] = f1

    return avg_loss, val_metrics

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    # Paths to your datasets (updated relative paths)
    script_dir = Path(__file__).parent.resolve()  # Directory where the script is located
    print(script_dir)

    # Update the paths to reflect the correct locations
    TRAIN_CSV = script_dir / '../../data/train_data/men_tshirt_train.csv'
    VAL_CSV = script_dir / '../../data/val_data/men_tshirt_val.csv'

    # Verify that the CSV files exist
    if not TRAIN_CSV.is_file():
        logger.error(f"Training CSV file not found at {TRAIN_CSV}")
        raise FileNotFoundError(f"Training CSV file not found at {TRAIN_CSV}")
    if not VAL_CSV.is_file():
        logger.error(f"Validation CSV file not found at {VAL_CSV}")
        raise FileNotFoundError(f"Validation CSV file not found at {VAL_CSV}")

    # Print the paths for debugging
    print(f"TRAIN_CSV path: {TRAIN_CSV}")
    print(f"VAL_CSV path: {VAL_CSV}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Enhanced augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Edge Enhancer Parameters (customizable)
    edge_enhancer_params = {
        'sigma': 1.5,                  # Adjust for edge sensitivity
        'low_threshold': 0.05,         # Lower for more edges
        'high_threshold': 0.2,         # Adjust for edge confidence
        'enhancement_strength': 0.4,   # Lower for subtler enhancement
        'use_multiscale': True          # Enable multiscale detection
    }

    # Load combined embeddings from JSON (updated relative path)
    combined_embeddings_path = combined_embeddings_path = script_dir / 'word_embeddings/men_tshirt_embeddings.json'


    if not combined_embeddings_path.is_file():
        logger.error(f"Combined embeddings file not found at {combined_embeddings_path}")
        raise FileNotFoundError(f"Combined embeddings file not found at {combined_embeddings_path}")

    # Print the embeddings path for debugging
    print(f"Combined embeddings path: {combined_embeddings_path}")

    with open(combined_embeddings_path, 'r') as f:
        combined_embeddings = json.load(f)

    # Update attribute_names based on available embeddings
    attribute_names = list(combined_embeddings.keys())

    # Datasets and DataLoaders
    train_dataset = CustomWomenTShirtDataset(
        csv_path=TRAIN_CSV,
        transform=train_transforms,
        combined_embeddings=combined_embeddings,
        attribute_names=attribute_names,
        edge_enhancement=True,             # Enable edge enhancement for training
        edge_enhancer_params=edge_enhancer_params
    )
    val_dataset = CustomWomenTShirtDataset(
        csv_path=VAL_CSV,
        transform=val_transforms,
        combined_embeddings=combined_embeddings,
        attribute_names=attribute_names,
        edge_enhancement=True,             # Enable edge enhancement for validation
        edge_enhancer_params=edge_enhancer_params
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize two ConvNeXt models with different pretraining or configurations if desired
    image_model1 = timm.create_model("convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0)  # No classification head
    image_model2 = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=True, num_classes=0)   # Different ConvNeXt variant

    image_model1.to(device)
    image_model2.to(device)

    # Combined model with two ConvNeXt models, ResNetFeatureExtractor, and Enhanced Classification Heads
    model = EnhancedCombinedModel(
        image_model1=image_model1,
        image_model2=image_model2,
        combined_embeddings=combined_embeddings,
        dropout_prob=0.4,
        mlp_dim=512,
        normalization='layernorm'  # Choose 'layernorm' or 'batchnorm'
    )
    model.to(device)

    # Initialize weights for all attributes
    initial_weights = {attr: 1.0 for attr in model.attribute_names}

    # Initialize loss with initial weights
    criterion = WeightedCrossEntropyLoss(attribute_weights=initial_weights)

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    num_epochs = 40
    best_val_f1 = 0.0  # For checkpointing
    current_weights = initial_weights.copy()
    weight_adjustment_epoch = 3  # Adjust weights every 3 epochs

    # Ensure the save directory exists (updated relative path)
    save_dir = script_dir / '../../men-tshirt/models'
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")

        # Training Phase
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validation Phase
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        # Adjust weights every few epochs
        if (epoch + 1) % weight_adjustment_epoch == 0:
            new_weights = adjust_weights(val_metrics, current_weights)
            current_weights = new_weights
            criterion.attribute_weights = new_weights

            logger.info("Updated attribute weights:")
            for attr, weight in new_weights.items():
                logger.info(f"{attr}: {weight:.4f}")

        # Calculate average metrics
        avg_train_f1 = np.mean(list(train_metrics.values()))
        avg_val_f1 = np.mean(list(val_metrics.values()))

        # Log metrics
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Avg Train F1: {avg_train_f1:.4f}, "
            f"Avg Val F1: {avg_val_f1:.4f}"
        )

        # Log individual attribute metrics
        for attr in model.attribute_names:
            train_f1 = train_metrics.get(attr, 0.0)
            val_f1 = val_metrics.get(attr, 0.0)
            logger.info(
                f"{attr} - Train F1: {train_f1:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )

        # Save best model based on average validation F1
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'attribute_weights': current_weights,
                'val_f1': best_val_f1
            }, save_dir / 'men_tshirt_visual_text_combined_best.pth')
            logger.info(f"New best model saved with Avg Val F1: {best_val_f1:.4f}")

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'attribute_weights': current_weights,
        'val_f1': avg_val_f1
    }, save_dir / 'men_tshirt_visual_text_combined_final.pth')
    logger.info("Training completed. Final model saved.")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
