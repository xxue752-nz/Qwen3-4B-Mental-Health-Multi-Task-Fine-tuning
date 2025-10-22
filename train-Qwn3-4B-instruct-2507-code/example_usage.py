#!/usr/bin/env python3
"""
Example usage of Qwen3-4B Mental Health Multi-Task Fine-tuning Framework
"""

import os
import sys
import logging
from qwen3_lora_multitask_weighted_optimized import (
    Qwen3LoRAMultiTaskTrainer, MultiTaskConfig, create_optimized_task_configs
)
from improved_logprob_implementation import create_improved_logprob_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_training_example():
    """Basic training example without Log-Prob evaluation"""
    logger.info("🚀 Starting basic training example...")
    
    # Create task configurations
    tasks = create_optimized_task_configs()
    
    # LoRA configuration
    lora_config = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # Create multi-task configuration
    config = MultiTaskConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        use_weighted_loss=True,
        output_dir="./qwen3_trained_model_basic",
        lora_r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"]
    )
    
    # Create trainer and start training
    trainer = Qwen3LoRAMultiTaskTrainer(config, tasks, lora_config)
    trainer.train()
    
    logger.info("✅ Basic training completed!")

def advanced_logprob_training_example():
    """Advanced training example with Log-Prob evaluation"""
    logger.info("🚀 Starting advanced Log-Prob training example...")
    
    # Create task configurations
    tasks = create_optimized_task_configs()
    
    # LoRA configuration
    lora_config = {
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # Create multi-task configuration
    config = MultiTaskConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        use_weighted_loss=True,
        output_dir="./qwen3_trained_model_logprob",
        lora_r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"]
    )
    
    # Create improved Log-Prob trainer
    Qwen3LoRAMultiTaskTrainerWithLogProb = create_improved_logprob_trainer(Qwen3LoRAMultiTaskTrainer)
    
    # Initialize trainer with Log-Prob parameters
    trainer = Qwen3LoRAMultiTaskTrainerWithLogProb(
        config, tasks, lora_config, 
        alpha=5.0,  # sigmoid sharpness
        beta=0.3    # CE vs BACC trade-off
    )
    
    # Train the model
    trainer.train()
    
    logger.info("✅ Advanced Log-Prob training completed!")

def evaluation_example():
    """Example of evaluating a trained model"""
    logger.info("🔍 Starting evaluation example...")
    
    # This would typically load a pre-trained model
    # For demonstration, we'll show the structure
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Example: Load test data for evaluation
    # test_data = pd.read_csv("data/your_test_data.csv")
    
    # Example: Evaluate with Log-Prob method
    # metrics = trainer.evaluate_task_with_logprob(task_config, test_data)
    # print(f"Evaluation metrics: {metrics}")
    
    logger.info("✅ Evaluation example structure shown!")

def main():
    """Main function to run examples"""
    print("🎯 Qwen3-4B Mental Health Multi-Task Fine-tuning Examples")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        logger.warning("⚠️  Data directory not found. Please create 'data/' directory and add your datasets.")
        logger.info("Expected data files:")
        logger.info("  - data/dreaddit_StressAnalysis - Sheet1.csv")
        logger.info("  - data/Reddit_depression_dataset.csv")
        logger.info("  - data/SDCNL.csv")
        logger.info("  - data/500_Reddit_users_posts_labels.csv")
        return
    
    # Check if CUDA is available
    import torch
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available. Training will be slow on CPU.")
    
    # Run examples
    try:
        # Uncomment the example you want to run
        # basic_training_example()
        # advanced_logprob_training_example()
        # evaluation_example()
        
        logger.info("📝 Examples ready! Uncomment the desired example in main() to run.")
        logger.info("💡 Start with basic_training_example() for simple training")
        logger.info("💡 Use advanced_logprob_training_example() for Log-Prob evaluation")
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        logger.info("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
