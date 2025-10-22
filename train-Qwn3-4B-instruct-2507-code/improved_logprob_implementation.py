#!/usr/bin/env python3
"""
Improved Log-Prob implementation, strictly following document formulas
Includes standard Log-Prob scoring and BACC surrogate loss
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImprovedLogProbEvaluator:
    """Improved Log-Prob evaluator, implementing standard formulas"""
    
    def __init__(self, model, tokenizer, alpha: float = 5.0, beta: float = 0.3):
        """
        Initialize Log-Prob evaluator
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            alpha: sigmoid sharpness parameter (1-20)
            beta: CE vs BACC trade-off weight (0.1-0.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha  # sigmoid sharpness
        self.beta = beta    # CE vs BACC trade-off
        
    @torch.no_grad()
    def predict_logits(self, prompts: List[str], label_tokens: Tuple[str, ...] = ("0", "1"), 
                      max_len: int = 1024) -> List[int]:
        """
        Standard Log-Prob scoring method
        Calculate log p(0|prompt) vs log p(1|prompt) and choose the higher one
        
        Args:
            prompts: Input prompt list
            label_tokens: Candidate label tokens (e.g., ("0", "1") or ("0", "1", "2", "3"))
            max_len: Maximum sequence length
            
        Returns:
            Predicted label list
        """
        # Ensure single token labels, otherwise sum over multiple tokens
        label_ids = []
        for token in label_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                label_ids.append(ids[0])
            else:
                # Multi-token case: use first token or special handling
                label_ids.append(ids[0])
                logger.warning(f"Multi-token label '{token}' detected, using first token")
        
        preds = []
        for prompt in prompts:
            # Encode prompt
            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            
            # Forward pass
            out = self.model(**inp, use_cache=False)
            
            # Get logits of the last token
            logits = out.logits[:, -1, :]  # [1, vocab_size]
            
            # Calculate log-probability for each label
            log_probs = []
            for label_id in label_ids:
                lp = torch.log_softmax(logits, dim=-1)[0, label_id].item()
                log_probs.append(lp)
            
            # Choose the label with highest log-prob
            pred = np.argmax(log_probs)
            preds.append(pred)
        
        return preds
    
    def compute_bacc_surrogate_loss(self, logits: torch.Tensor, true_labels: torch.Tensor, 
                                  task_classes: int, gamma_c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute BACC surrogate loss
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            true_labels: True labels [batch_size]
            task_classes: Number of task classes
            gamma_c: Class weights, if None use uniform weights
            
        Returns:
            BACC surrogate loss
        """
        batch_size = logits.size(0)
        
        # If no class weights provided, use uniform weights
        if gamma_c is None:
            gamma_c = torch.ones(task_classes, device=logits.device)
        
        # Step 1: Calculate margin for each class
        # m_{i,c} = z_{i,c} - log(sum_{k!=c} e^(z_{i,k}))
        margins = torch.zeros(batch_size, task_classes, device=logits.device)
        
        for c in range(task_classes):
            # For class c, calculate margin
            z_c = logits[:, c]  # [batch_size]
            
            # Calculate log-sum-exp of other classes
            other_logits = torch.cat([logits[:, :c], logits[:, c+1:]], dim=1)
            log_sum_exp_others = torch.logsumexp(other_logits, dim=1)
            
            margins[:, c] = z_c - log_sum_exp_others
        
        # Step 2: Calculate soft "is correct" score
        # s_{i,c} = sigma(alpha * m_{i,c})
        sigmoid_scores = torch.sigmoid(self.alpha * margins)  # [batch_size, num_classes]
        
        # Step 3: Calculate TPR for each class
        # TPR_c = (1 / |I_c|) * sum_{i in I_c} s_{i,c}
        tpr_c = torch.zeros(task_classes, device=logits.device)
        
        for c in range(task_classes):
            # Find sample indices with true label c
            mask_c = (true_labels == c)
            if mask_c.sum() > 0:
                tpr_c[c] = sigmoid_scores[mask_c, c].mean()
            else:
                tpr_c[c] = 0.0  # If no samples belong to class c
        
        # Step 4: Calculate BACC surrogate loss
        # LBACC = 1 - (1 / sum_c gamma_c) * sum_c gamma_c * TPR_c
        gamma_sum = gamma_c.sum()
        if gamma_sum > 0:
            weighted_tpr = (gamma_c * tpr_c).sum()
            bacc_loss = 1.0 - weighted_tpr / gamma_sum
        else:
            bacc_loss = torch.tensor(1.0, device=logits.device)
        
        return bacc_loss
    
    def compute_combined_loss(self, logits: torch.Tensor, true_labels: torch.Tensor, 
                            task_classes: int, gamma_c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined loss: L = LCE + β * LBACC
        
        Args:
            logits: Model output logits
            true_labels: True labels
            task_classes: Number of task classes
            gamma_c: Class weights
            
        Returns:
            Combined loss
        """
        # Step 1: Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, true_labels)
        
        # Step 2: BACC Surrogate Loss
        bacc_loss = self.compute_bacc_surrogate_loss(logits, true_labels, task_classes, gamma_c)
        
        # Step 3: Combined loss
        combined_loss = ce_loss + self.beta * bacc_loss
        
        return combined_loss
    
    def evaluate_task_with_improved_logprob(self, task_config, test_data, 
                                          label_tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate task using improved Log-Prob method
        
        Args:
            task_config: Task configuration
            test_data: Test data
            label_tokens: Label tokens, if None use task_config.class_names
            
        Returns:
            Evaluation metrics dictionary
        """
        if label_tokens is None:
            label_tokens = [str(x) for x in task_config.class_names]
        
        logger.info(f"Evaluating task using improved Log-Prob method: {task_config.name}")
        logger.info(f"Label tokens: {label_tokens}")
        
        # Prepare data
        prompts = []
        true_labels = []
        
        for _, row in test_data.iterrows():
            text = str(row[task_config.text_column])
            label = str(row[task_config.label_column])
            
            # Format prompt
            prompt = task_config.prompt_template.format(text=text, label="")
            prompts.append(prompt)
            true_labels.append(label)
        
        # Use improved Log-Prob prediction
        predictions = self.predict_logits(prompts, tuple(label_tokens))
        
        # Convert to numeric labels
        label_map = {token: i for i, token in enumerate(label_tokens)}
        y_true = [label_map.get(str(t), 0) for t in true_labels]
        y_pred = predictions
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        
        logger.info(f"Task {task_config.name} improved Log-Prob results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics

def create_improved_logprob_trainer(base_trainer_class):
    """Create improved Log-Prob trainer class"""
    
    class ImprovedLogProbTrainer(base_trainer_class):
        """Improved Log-Prob trainer"""
        
        def __init__(self, *args, alpha: float = 5.0, beta: float = 0.3, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.beta = beta
            # Initialize logprob evaluator after model is set up
            self.logprob_evaluator = None
            
        def _setup_logprob_evaluator(self):
            """Setup Log-Prob evaluator after model is initialized"""
            if self.logprob_evaluator is None:
                # Check if we have access to model and tokenizer through different paths
                model_access = None
                tokenizer_access = None
                
                # Try direct access
                if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
                    model_access = self.model
                    tokenizer_access = self.tokenizer
                # Try through internal trainer
                elif hasattr(self, '_trainer') and hasattr(self._trainer, 'model') and hasattr(self._trainer, 'tokenizer'):
                    model_access = self._trainer.model
                    tokenizer_access = self._trainer.tokenizer
                
                if model_access is not None and tokenizer_access is not None:
                    self.logprob_evaluator = ImprovedLogProbEvaluator(
                        model_access, tokenizer_access, self.alpha, self.beta
                    )
                else:
                    logger.warning("Cannot setup LogProb evaluator: model or tokenizer not accessible")
        
        def compute_loss(self, model, inputs, return_outputs=False):
            """Override loss computation, using BACC surrogate loss"""
            # Setup evaluator if not already done
            self._setup_logprob_evaluator()
            
            labels = inputs.get("labels")
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            if labels is not None and self.logprob_evaluator is not None:
                # Calculate combined loss
                loss = self.logprob_evaluator.compute_combined_loss(
                    logits, labels, 
                    task_classes=logits.size(-1)
                )
            else:
                loss = outputs.get("loss", torch.tensor(0.0))
            
            return (loss, outputs) if return_outputs else loss
        
        def evaluate_task_with_logprob(self, task_config, test_data):
            """Evaluate task using improved Log-Prob method"""
            # Setup evaluator if not already done
            self._setup_logprob_evaluator()
            
            if self.logprob_evaluator is None:
                raise RuntimeError("LogProb evaluator not initialized. Make sure model and tokenizer are set up.")
                
            return self.logprob_evaluator.evaluate_task_with_improved_logprob(
                task_config, test_data
            )
    
    return ImprovedLogProbTrainer

# Usage example
if __name__ == "__main__":
    # Example: How to use improved Log-Prob evaluator
    print("Improved Log-Prob implementation is ready")
    print("Main features:")
    print("1. Standard Log-Prob scoring: log p(0|prompt) vs log p(1|prompt)")
    print("2. BACC surrogate loss: Balanced accuracy surrogate loss")
    print("3. Combined loss: L = LCE + β * LBACC")
    print("4. Tunable parameters: α (sharpness), β (trade-off)")