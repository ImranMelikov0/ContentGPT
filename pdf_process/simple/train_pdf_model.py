from pdf_process.simple.pdf_processor import PDFProcessor
from pdf_process.simple.model import ModelTrainer
import logging


def main():
    # Configuration
    config = {
        'max_length': 128,
        'batch_size': 4,
        'epochs': 50,
        'patience': 5,
        'learning_rate': 5e-5,
        'max_grad_norm': 1.0,
        'min_text_length': 10,
        'warmup_steps': 1000,
        'weight_decay': 0.01
    }

    try:
        # PDF processing
        pdf_processor = PDFProcessor(min_line_length=10)
        pdf_processor.process_pdf_directory(
            pdf_dir='../../pdfs',
            output_file='processed_data/processed_text.txt'
        )

        # Model training
        trainer = ModelTrainer(config)
        trainer.setup_model()

        # Dataset preparation
        dataset = trainer.prepare_dataset('processed_data/processed_text.txt')

        # Training
        history = trainer.train(dataset)
        logging.info(f"Training completed. Best loss: {history['loss']:.4f}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
