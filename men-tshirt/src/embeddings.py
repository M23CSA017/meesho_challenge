from transformers import RobertaModel, RobertaTokenizer
import torch
import json
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClothingEmbeddingGenerator:
    def __init__(self, category: str, attributes: Dict[str, str], device: str = None):
        self.category = category
        self.attributes = attributes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing embedding generator for {category} on {self.device}")
        
        # Load models
        self.roberta_tokenizer, self.roberta_model = self._setup_roberta()
        self.sentence_transformer = self._setup_sentence_transformer()
        
    def _setup_roberta(self):
        """Initialize RoBERTa model with category-specific tokens"""
        try:
            model_name = "roberta-base"
            tokenizer = RobertaTokenizer.from_pretrained(
                model_name,
                model_max_length=64,
                padding_side='right',
                truncation_side='right'
            )
            
            # Add category-specific tokens
            special_tokens = {
                'additional_special_tokens': [
                    f'[{attr_name.upper()}]' for attr_name in self.attributes.values()
                ]
            }
            tokenizer.add_special_tokens(special_tokens)
            
            model = RobertaModel.from_pretrained(model_name)
            model.resize_token_embeddings(len(tokenizer))
            model.to(self.device)
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error setting up RoBERTa: {e}")
            raise

    def _setup_sentence_transformer(self):
        """Initialize SentenceTransformer model"""
        try:
            return SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        except Exception as e:
            logger.error(f"Error setting up SentenceTransformer: {e}")
            raise

    def _generate_roberta_embeddings(self, attribute_values: Dict[str, List[str]], batch_size: int = 32):
        """Generate RoBERTa embeddings with category-specific prompting"""
        embeddings = {}
        
        with torch.no_grad():
            for attr, values in attribute_values.items():
                if not values:  # Skip empty attribute lists
                    continue
                    
                embeddings[attr] = []
                attr_name = self.attributes[attr]  # Map attr_# to actual attribute name
                
                for i in range(0, len(values), batch_size):
                    batch_values = values[i:i + batch_size]
                    # Category-specific prompting
                    texts = [
                        f"This {self.category} has a {attr_name} of {value}."
                        for value in batch_values
                    ]
                    
                    try:
                        inputs = self.roberta_tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=64,
                            return_attention_mask=True
                        ).to(self.device)
                        
                        outputs = self.roberta_model(**inputs)
                        embeddings_weighted = (outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1)).sum(1)
                        embeddings_weighted = embeddings_weighted / inputs['attention_mask'].sum(-1, keepdim=True)
                        normalized_embeddings = F.normalize(embeddings_weighted, p=2, dim=1)
                        
                        for value, embedding in zip(batch_values, normalized_embeddings):
                            embeddings[attr].append({
                                'value': value,
                                'roberta_embedding': embedding.cpu().numpy(),
                                'attribute': attr_name
                            })
                            
                    except Exception as e:
                        logger.error(f"Error generating RoBERTa embeddings for {attr}: {e}")
                        continue
                        
        return embeddings

    def _generate_sentence_transformer_embeddings(self, attribute_values: Dict[str, List[str]], batch_size: int = 32):
        """Generate SentenceTransformer embeddings with category-specific context"""
        embeddings = {}
        
        for attr, values in attribute_values.items():
            if not values:  # Skip empty attribute lists
                continue
                
            try:
                embeddings[attr] = []
                attr_name = self.attributes[attr]  # Map attr_# to actual attribute name
                
                texts = [
                    f"The {self.category}'s {attr_name} feature is {value}, which defines its {attr_name} characteristic"
                    for value in values
                ]
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_values = values[i:i + batch_size]
                    
                    batch_embeddings = self.sentence_transformer.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        batch_size=batch_size
                    )
                    
                    for value, embedding in zip(batch_values, batch_embeddings):
                        embeddings[attr].append({
                            'value': value,
                            'sentence_transformer_embedding': embedding.cpu().numpy(),
                            'attribute': attr_name
                        })
                        
            except Exception as e:
                logger.error(f"Error generating SentenceTransformer embeddings for {attr}: {e}")
                continue
                
        return embeddings

    def generate_combined_embeddings(self, attribute_values: Dict[str, List[str]], output_path: str):
        """Generate and combine both types of embeddings"""
        try:
            logger.info(f"Generating embeddings for {self.category}...")
            roberta_embeddings = self._generate_roberta_embeddings(attribute_values)
            sentence_embeddings = self._generate_sentence_transformer_embeddings(attribute_values)
            
            # Combine embeddings
            combined_embeddings = {}
            for attr in attribute_values.keys():
                if not attribute_values[attr]:  # Skip empty attributes
                    continue
                    
                combined_embeddings[attr] = []
                
                for roberta_entry in roberta_embeddings[attr]:
                    value = roberta_entry['value']
                    sentence_entry = next(
                        (e for e in sentence_embeddings[attr] if e['value'] == value),
                        None
                    )
                    
                    if sentence_entry:
                        combined_embeddings[attr].append({
                            'value': value,
                            'roberta_embedding': roberta_entry['roberta_embedding'].tolist(),
                            'sentence_transformer_embedding': sentence_entry['sentence_transformer_embedding'].tolist(),
                            'attribute': roberta_entry['attribute']
                        })
            
            # Save embeddings
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(combined_embeddings, f, indent=4)
                
            logger.info(f"Combined embeddings saved to {output_path}")
            return combined_embeddings
            
        except Exception as e:
            logger.error(f"Error generating combined embeddings: {e}")
            raise

# Usage example for Kurtis
if __name__ == "__main__":
    # Define attributes and values for the 'Kurtis' category
    category = 'Men Tshirts'
    attributes = {
        "attr_1": "color",
        "attr_2": "neck",
        "attr_3": "pattern",
        "attr_4": "print_or_pattern_type",
        "attr_5": "sleeve_length",
    }
    attribute_values = {
        "attr_1": ["default", "multicolor", "black", "white"],
        "attr_2": ["round", "polo"],
        "attr_3": ["printed", "solid"],
        "attr_4": ["default", "solid", "typography"],
        "attr_5": ["short sleeves", "long sleeves"],
    }
    output_path = "word_embeddings/men_tshirt_embeddings.json"
    
    # Initialize the generator
    generator = ClothingEmbeddingGenerator(category, attributes)
    # Generate embeddings
    generator.generate_combined_embeddings(attribute_values, output_path)
