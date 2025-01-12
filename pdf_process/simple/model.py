import logging
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
from pathlib import Path
import json
from typing import Dict
from datetime import datetime


class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_paths()
        self.setup_device()
        self.tokenizer = None
        self.model = None

    def setup_paths(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f'models/run_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

    def setup_device(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU is active: {len(gpus)} available")
                self.device = "GPU"
            else:
                logging.info("Using CPU")
                self.device = "CPU"
        except Exception as e:
            logging.error(f"Hardware error: {str(e)}")
            self.device = "CPU"

    def setup_model(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

            model_config = GPT2Config(
                vocab_size=len(self.tokenizer),
                n_positions=self.config['max_length'],
                n_ctx=self.config['max_length'],
                n_embd=768,
                n_layer=6,
                n_head=12
            )

            self.model = TFGPT2LMHeadModel(model_config)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            )
            self.model.compile(optimizer=optimizer)

            logging.info("Model successfully created")

        except Exception as e:
            logging.error(f"Model error: {str(e)}")
            raise

    def prepare_dataset(self, text_path: str):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Split text into smaller chunks
            chunk_size = self.config['max_length'] * 20
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

            all_input_ids = []
            all_attention_masks = []

            for chunk in text_chunks:
                encodings = self.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.config['max_length'],
                    padding='max_length',
                    return_tensors='tf'
                )

                all_input_ids.append(encodings['input_ids'])
                all_attention_masks.append(encodings['attention_mask'])

            # Combine all chunks
            input_ids = tf.concat(all_input_ids, axis=0)
            attention_masks = tf.concat(all_attention_masks, axis=0)

            dataset = tf.data.Dataset.from_tensor_slices({
                'input_ids': input_ids,
                'attention_mask': attention_masks,
                'labels': input_ids
            })

            # Show dataset size
            total_samples = tf.data.experimental.cardinality(dataset).numpy()
            logging.info(f"Total sample count: {total_samples}")

            # Larger shuffle buffer
            dataset = dataset.shuffle(10000)
            dataset = dataset.batch(self.config['batch_size'])
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

        except Exception as e:
            logging.error(f"Dataset error: {str(e)}")
            raise

    def train(self, dataset):
        try:
            best_loss = float('inf')
            patience_counter = 0
            history = []

            for epoch in range(self.config['epochs']):
                logging.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
                total_loss = 0
                num_batches = 0

                for batch in dataset:
                    with tf.GradientTape() as tape:
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                            training=True
                        )
                        loss = outputs.loss

                    # Add gradient clipping
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    gradients, _ = tf.clip_by_global_norm(gradients, self.config['max_grad_norm'])
                    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    batch_loss = float(loss)
                    total_loss += batch_loss
                    num_batches += 1

                    if num_batches % 5 == 0:
                        logging.info(f"Batch {num_batches} - Loss: {batch_loss:.4f}")

                avg_loss = total_loss / num_batches
                history.append(avg_loss)
                logging.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

                # Check loss change
                if len(history) > 1:
                    loss_change = abs(history[-1] - history[-2])
                    logging.info(f"Loss change: {loss_change:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.model.save_pretrained(self.output_dir / 'best_model')
                    self.tokenizer.save_pretrained(self.output_dir / 'best_model')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        logging.info(f"Early stopping - No improvement for {self.config['patience']} epochs")
                        break

            return {'loss': best_loss, 'history': history}

        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise
