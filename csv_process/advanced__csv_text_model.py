import pandas as pd
import tensorflow as tf
import re
import logging
import os
from datetime import datetime
from transformers import GPT2Tokenizer
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TextGenerator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.setup_paths()

    def setup_paths(self):
        """Creates folders for model and log files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = os.path.join('models', f'model_{timestamp}')
        self.log_path = os.path.join('logs', f'run_{timestamp}')

        for path in [self.model_path, self.log_path]:
            os.makedirs(path, exist_ok=True)

    def load_data(self):
        """Loads and preprocesses the data"""
        logging.info("Loading data...")
        try:
            df = pd.read_csv(self.config['data_path'])
            texts = df[self.config['text_column']].astype(str).tolist()
            texts = [self.clean_text(text) for text in texts]
            logging.info(f"Total {len(texts)} texts loaded")
            return texts
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            raise

    @staticmethod
    def clean_text(text):
        """Cleans and normalizes the text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def setup_tokenizer(self):
        """Prepares the tokenizer"""
        logging.info("Loading tokenizer...")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='../model_cache')
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Save the tokenizer
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info("Tokenizer successfully loaded and saved!")
            return self.tokenizer
        except Exception as e:
            logging.error(f"Tokenizer loading error: {str(e)}")
            raise

    def create_transformer_block(self, inputs, d_model, num_heads, dff, rate=0.1):
        """Creates a transformer block"""
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(inputs, inputs)
        attention_output = Dropout(rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

        ffn_output = Dense(dff, activation='relu')(attention_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

    def build_model(self):
        """Builds the model architecture"""
        logging.info("Building model...")
        try:
            # Input layer
            inputs = tf.keras.layers.Input(shape=(self.config['max_length'],))

            # Embedding layer
            x = tf.keras.layers.Embedding(self.tokenizer.vocab_size, self.config['d_model'])(inputs)

            # LSTM layers
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512))(x)
            x = tf.keras.layers.Dropout(0.2)(x)

            # Dense layers
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)

            # Output layer
            outputs = tf.keras.layers.Dense(self.tokenizer.vocab_size, activation='softmax')(x)

            # Create the model
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Optimizer and loss
            optimizer = Adam(
                learning_rate=self.config['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )

            self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logging.info("Model architecture successfully built!")
            logging.info("\nModel summary:")
            self.model.summary()
            return self.model

        except Exception as e:
            logging.error(f"Model building error: {str(e)}")
            raise

    def prepare_data(self, texts):
        """Prepares the data for training"""
        logging.info("Preparing data...")
        try:
            # Tokenization
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.config['max_length'],
                return_tensors='tf'
            )

            # Create input and target sequences
            input_ids = encodings['input_ids']
            target_ids = tf.roll(input_ids, -1, axis=1)

            # Flatten the targets
            target_ids = tf.reshape(target_ids, [-1, 1])  # One target per example

            # Shuffle the dataset
            dataset_size = tf.shape(input_ids)[0].numpy()
            indices = tf.random.shuffle(tf.range(dataset_size))
            input_ids = tf.gather(input_ids, indices)
            target_ids = tf.gather(target_ids, indices)

            # Train-validation split
            val_size = int(dataset_size * self.config['validation_split'])

            X_train = input_ids[val_size:]
            X_val = input_ids[:val_size]
            y_train = target_ids[val_size:]
            y_val = target_ids[:val_size]

            logging.info(f"Training set size: {tf.shape(X_train)[0].numpy()}")
            logging.info(f"Validation set size: {tf.shape(X_val)[0].numpy()}")
            logging.info(f"Input shape: {X_train.shape}, Target shape: {y_train.shape}")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            logging.error(f"Data preparation error: {str(e)}")
            raise

    def train(self, X_train, X_val, y_train, y_val):
        """Trains the model"""
        logging.info("Model training started...")

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.model_path, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(log_dir=self.log_path),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]

        # Training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        return history

    def generate_text(self, prompt, max_length=50, temperature=0.7):
        """Generates text"""
        try:
            # Tokenize the prompt and apply padding
            input_ids = self.tokenizer.encode(prompt, return_tensors='tf')
            input_ids = pad_sequences(input_ids, maxlen=self.config['max_length'], padding='pre')

            generated_text = prompt

            for _ in range(max_length):
                # Make a prediction
                predictions = self.model(input_ids)

                # Get the predictions for the last token and apply temperature
                last_token_predictions = predictions[0] / temperature

                # Apply top-p (nucleus) sampling
                p = 0.9
                sorted_indices = tf.argsort(last_token_predictions, direction='DESCENDING')
                sorted_probs = tf.gather(last_token_predictions, sorted_indices)
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_probs))
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove = tf.concat([[False], sorted_indices_to_remove[:-1]], axis=0)
                indices = tf.boolean_mask(sorted_indices, ~sorted_indices_to_remove)

                # Get the remaining probabilities
                filtered_probs = tf.boolean_mask(sorted_probs, ~sorted_indices_to_remove)
                filtered_probs = tf.nn.softmax(filtered_probs)

                # Choose the next token
                predicted_id = tf.random.categorical(
                    tf.math.log(tf.expand_dims(filtered_probs, 0)),
                    num_samples=1
                )[0, 0]
                predicted_id = indices[predicted_id]

                # If EOS token or punctuation appears, check
                if predicted_id == self.tokenizer.eos_token_id:
                    break

                # Decode the predicted token
                predicted_word = self.tokenizer.decode(predicted_id)

                # Recheck
                last_word = generated_text.split()[-1] if generated_text else ""
                if predicted_word.strip() == last_word:
                    continue

                # Control spaces
                if not predicted_word.startswith(' ') and not predicted_word.startswith(('.', ',', '!', '?')):
                    predicted_word = ' ' + predicted_word

                generated_text += predicted_word

                # Stop when sentences are complete
                if predicted_word.strip() in ['.', '!', '?']:
                    break

                # Prepare new input id and apply padding
                input_ids = self.tokenizer.encode(generated_text, return_tensors='tf')
                input_ids = pad_sequences(input_ids, maxlen=self.config['max_length'], padding='pre')

                # Control maximum length
                if len(generated_text.split()) >= max_length:
                    if not generated_text.strip().endswith(('.', '!', '?')):
                        generated_text += '.'
                    break

            # Last clear
            generated_text = ' '.join(generated_text.split())  # Clear extra spaces
            generated_text = generated_text.replace(' .', '.')
            generated_text = generated_text.replace(' ,', ',')
            generated_text = generated_text.replace(' !', '!')
            generated_text = generated_text.replace(' ?', '?')
            generated_text = generated_text.strip()

            return generated_text

        except Exception as e:
            logging.error(f"Text generate error: {str(e)}")
            return prompt

def main():
    # Model configration
    config = {
        'data_path': '{your_csv_file}',
        'text_column': 'Text',
        'max_length': 128,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 2e-5,
        'validation_split': 0.15,
        'early_stopping_patience': 5,
        'num_transformer_blocks': 3,
        'd_model': 768,
        'num_heads': 12,
        'dff': 3072,
        'dropout_rate': 0.1
    }

    try:
        # Generate and train model
        generator = TextGenerator(config)
        tokenizer = generator.setup_tokenizer()  # Generate and save tokenizer
        texts = generator.load_data()
        model = generator.build_model()

        X_train, X_val, y_train, y_val = generator.prepare_data(texts)

        # Train
        history = generator.train(X_train, X_val, y_train, y_val)


        if not os.path.exists('tokenizer.pickle'):
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(generator.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("Tokenizer saved!")

        # Test
        test_prompts = [
            "Please help me find",
            "I want to listen to",
            "Can you search for"
        ]

        logging.info("\nTest results:")
        for prompt in test_prompts:
            generated_text = generator.generate_text(prompt, max_length=50)
            logging.info(f"\nInput: {prompt}")
            logging.info(f"Output: {generated_text}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()