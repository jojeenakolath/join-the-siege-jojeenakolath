import asyncio
from pathlib import Path
from trainer_script import DocumentTrainer

async def main():
    # Initialize trainer
    trainer = DocumentTrainer(
        model_name="roberta-base",
        output_dir="models",
        batch_size=16,
        num_epochs=5
    )
    
    # Setup paths
    raw_data_dir = Path("training_data/")
    processed_dir = Path("processed_data")
    processed_dir.mkdir(exist_ok=True)
    
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    
    # Prepare training data
    try:
        print("Preparing training data...")
        train_path = await trainer.prepare_training_data(
            raw_data_dir=raw_data_dir,
            output_path=train_path
        )
        
        print("Training model...")
        metrics = trainer.train(
            train_path=train_path,
            val_path=val_path if val_path.exists() else None,
            experiment_name="document_classifier_v1"
        )
        
        print("Training complete!")
        print("Metrics:", metrics)
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    asyncio.run(main())