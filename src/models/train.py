# src/models/train.py
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import mlflow
import mlflow.pytorch
import json
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdCreativeTrainer:
    def __init__(
        self, 
        model_name: str = "t5-small",
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("ad-creative-generation")
    
    def load_training_data(self, data_path: str) -> Dataset:
        """Load and prepare training dataset"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        dataset = Dataset.from_dict({
            'input': [item['input'] for item in data],
            'target': [item['target'] for item in data]
        })
        
        return dataset
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize inputs and targets"""
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        labels = self.tokenizer(
            examples['target'],
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        output_dir: str = "models/checkpoints",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5
    ):
        """Train the model with MLflow tracking"""
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="none"  # We'll use MLflow
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"t5-ad-generation-{num_epochs}epochs"):
            # Log parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("train_samples", len(train_dataset))
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics['train_runtime'],
                "train_samples_per_second": train_result.metrics['train_samples_per_second']
            })
            
            # Save model
            model_path = f"{output_dir}/final_model"
            trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name="ad-creative-generator"
            )
            
            # Log artifacts
            mlflow.log_artifacts(output_dir, artifact_path="checkpoints")
            
            logger.info("Training completed!")
            logger.info(f"Model saved to {model_path}")
            
        return trainer
    
    def evaluate_generation(self, test_inputs: List[str]) -> Dict:
        """Evaluate model generation quality"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for input_text in test_inputs:
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )
                
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.8
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({
                    'input': input_text,
                    'generated': generated
                })
        
        return results

if __name__ == "__main__":
    # Initialize trainer
    trainer = AdCreativeTrainer(
        model_name="t5-small",
        mlflow_tracking_uri="http://localhost:5000"
    )
    
    # Load data
    dataset = trainer.load_training_data("data/processed/training_data.json")
    
    # Split dataset
    train_test = dataset.train_test_split(test_size=0.1)
    
    # Train
    trainer.train(
        train_dataset=train_test['train'],
        eval_dataset=train_test['test'],
        num_epochs=3,
        batch_size=8
    )
    
    # Test generation
    test_inputs = [
        "Generate ad for: Wireless Headphones. Category: Electronics. Description: Premium noise-cancelling wireless headphones with 30-hour battery life."
    ]
    results = trainer.evaluate_generation(test_inputs)
    print("\n--- Generation Test ---")
    for r in results:
        print(f"Input: {r['input']}")
        print(f"Generated: {r['generated']}\n")