# llm_prompt_recovery.py
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel, GPT2TokenizerFast
from keras_nlp.models import GemmaCausalLM 
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
   def __init__(self):
       self.BATCH_SIZE = 32
       self.LEARNING_RATE = 2e-5
       self.EPOCHS = 10
       self.MAX_LENGTH = 512
       self.MODEL_PATH = "google/gemma-7b-it"
       self.ENCODER_PATH = "sentence-transformers/sentence-t5-base"
       self.TRAIN_PATH = "data/train.csv"
       self.VAL_PATH = "data/val.csv" 
       self.TEST_PATH = "data/test.csv"
       self.CHECKPOINT_DIR = "checkpoints"
       self.WANDB_PROJECT = "llm-prompt-recovery"
       self.SEED = 42
       
       # Create dirs
       os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

class DataProcessor:
   def __init__(self, config):
       self.config = config
       self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
       self.encoder_tokenizer = GPT2TokenizerFast.from_pretrained(config.ENCODER_PATH)
       
   def load_and_preprocess_data(self, path):
       df = pd.read_csv(path)
       return self._preprocess_dataframe(df)
       
   def _preprocess_dataframe(self, df):
       processed = []
       for _, row in tqdm(df.iterrows(), total=len(df)):
           try:
               orig_tokens = self.tokenizer(
                   row['original_text'],
                   max_length=self.config.MAX_LENGTH,
                   padding='max_length',
                   truncation=True,
                   return_tensors='tf'
               )
               
               rewrite_tokens = self.tokenizer(
                   row['rewritten_text'],
                   max_length=self.config.MAX_LENGTH, 
                   padding='max_length',
                   truncation=True,
                   return_tensors='tf'
               )
               
               prompt_tokens = None
               if 'rewrite_prompt' in row:
                   prompt_tokens = self.tokenizer(
                       row['rewrite_prompt'],
                       max_length=self.config.MAX_LENGTH//4,
                       padding='max_length',
                       truncation=True,
                       return_tensors='tf'
                   )
               
               processed.append({
                   'id': row['id'],
                   'original': orig_tokens,
                   'rewritten': rewrite_tokens,
                   'prompt': prompt_tokens
               })
               
           except Exception as e:
               logger.error(f"Error processing row {row['id']}: {str(e)}")
               continue
               
       return processed

class PromptEngineering:
   def __init__(self):
       self.templates = {
           "style_transfer": [
               "Rewrite this text in the style of {author}",
               "Transform this passage to match {author}'s writing style",
               "Rewrite this as if {author} wrote it"
           ],
           "tone_shift": [
               "Rewrite this text to be more {tone}",
               "Make this text sound more {tone}",
               "Change the tone to be {tone}"
           ],
           "perspective": [
               "Rewrite this from {character}'s perspective",
               "Tell this story from {character}'s point of view",
               "Narrate this as {character}"
           ],
           "format_change": [
               "Rewrite this as a {format}",
               "Convert this into a {format}",
               "Transform this into {format} format"
           ]
       }
       
       self.style_params = {
           "author": ["Shakespeare", "Hemingway", "Jane Austen", "Dr. Seuss"],
           "tone": ["formal", "casual", "professional", "friendly"],
           "character": ["first person", "third person", "narrator"],
           "format": ["dialogue", "letter", "speech", "story"]
       }
   
   def generate_prompts(self, text):
       prompts = []
       for template_type, templates in self.templates.items():
           params = self.style_params[template_type.split('_')[0]]
           for template in templates:
               for param in params:
                   prompt = template.format(**{template_type.split('_')[0]: param})
                   prompts.append({
                       'original_text': text,
                       'rewrite_prompt': prompt
                   })
       return prompts

class PromptRecoveryModel(tf.keras.Model):
   def __init__(self, config):
       super().__init__()
       self.config = config
       
       # Base models
       self.gemma = GemmaCausalLM.from_pretrained(config.MODEL_PATH)
       self.sentence_encoder = AutoModel.from_pretrained(config.ENCODER_PATH)
       
       # LoRA config
       lora_config = LoraConfig(
           r=8,
           lora_alpha=16,
           target_modules=["q_proj", "v_proj"],
           lora_dropout=0.1,
           bias="none",
           task_type="CAUSAL_LM"
       )
       self.gemma = get_peft_model(self.gemma, lora_config)
       
       # Additional layers
       self.encoder_projection = tf.keras.layers.Dense(768, activation='gelu')
       
       self.cross_attention = tf.keras.layers.MultiHeadAttention(
           num_heads=8,
           key_dim=64,
           dropout=0.1
       )
       
       self.feed_forward = tf.keras.Sequential([
           tf.keras.layers.Dense(3072, activation='gelu'),
           tf.keras.layers.Dropout(0.1),
           tf.keras.layers.Dense(768)
       ])
       
       self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
       self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
       
       self.prompt_decoder = tf.keras.layers.Dense(
           self.gemma.config.vocab_size,
           activation='softmax'
       )

   def encode_pair(self, original, rewritten):
       # Encode both texts
       orig_encoding = self.sentence_encoder(original)[0]
       rewrite_encoding = self.sentence_encoder(rewritten)[0]
       
       # Project to same space
       orig_proj = self.encoder_projection(orig_encoding)
       rewrite_proj = self.encoder_projection(rewrite_encoding)
       
       # Concatenate
       return tf.concat([orig_proj, rewrite_proj], axis=-1)

   def call(self, inputs, training=False):
       # Encode input pair
       encoded = self.encode_pair(inputs['original'], inputs['rewritten'])
       
       # Multi-head attention
       attn_output = self.cross_attention(
           query=encoded,
           key=encoded,
           value=encoded,
           training=training
       )
       out1 = self.layer_norm1(encoded + attn_output)
       
       # Feed forward
       ff_output = self.feed_forward(out1)
       out2 = self.layer_norm2(out1 + ff_output)
       
       # Generate prompt
       return self.prompt_decoder(out2)

class TrainingPipeline:
   def __init__(self, config):
       self.config = config
       self.data_processor = DataProcessor(config)
       self.model = PromptRecoveryModel(config)
       self.prompt_engineer = PromptEngineering()
       
       # Initialize wandb
       wandb.init(project=config.WANDB_PROJECT)
       
   def train(self):
       # Load data
       train_data = self.data_processor.load_and_preprocess_data(self.config.TRAIN_PATH)
       val_data = self.data_processor.load_and_preprocess_data(self.config.VAL_PATH)
       
       # Create datasets
       train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
       train_dataset = train_dataset.shuffle(1000).batch(self.config.BATCH_SIZE)
       
       val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
       val_dataset = val_dataset.batch(self.config.BATCH_SIZE)
       
       # Compile model
       self.model.compile(
           optimizer=tf.keras.optimizers.Adam(self.config.LEARNING_RATE),
           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
           metrics=['accuracy']
       )
       
       # Callbacks
       callbacks = [
           tf.keras.callbacks.ModelCheckpoint(
               filepath=os.path.join(self.config.CHECKPOINT_DIR, 'model_{epoch:02d}.h5'),
               save_best_only=True,
               monitor='val_loss'
           ),
           tf.keras.callbacks.EarlyStopping(
               monitor='val_loss',
               patience=3
           ),
           wandb.keras.WandbCallback()
       ]
       
       # Train
       history = self.model.fit(
           train_dataset,
           validation_data=val_dataset,
           epochs=self.config.EPOCHS,
           callbacks=callbacks
       )
       
       return history
       
   def predict(self, test_path):
       # Load test data
       test_data = self.data_processor.load_and_preprocess_data(test_path)
       test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(self.config.BATCH_SIZE)
       
       # Generate predictions
       predictions = self.model.predict(test_dataset)
       
       # Convert to prompts
       predicted_prompts = []
       for pred in predictions:
           tokens = np.argmax(pred, axis=-1)
           prompt = self.data_processor.tokenizer.decode(tokens)
           predicted_prompts.append(prompt)
           
       # Create submission
       submission = pd.DataFrame({
           'id': [x['id'] for x in test_data],
           'rewrite_prompt': predicted_prompts
       })
       
       return submission

def main():
   # Set random seeds
   tf.random.set_seed(Config.SEED)
   np.random.seed(Config.SEED)
   
   # Initialize
   config = Config()
   pipeline = TrainingPipeline(config)
   
   # Train
   logger.info("Starting training...")
   history = pipeline.train()
   
   # Generate predictions
   logger.info("Generating predictions...")
   submission = pipeline.predict(config.TEST_PATH)
   
   # Save submission
   submission.to_csv('submission.csv', index=False)
   logger.info("Saved submission file")

if __name__ == "__main__":
   main()