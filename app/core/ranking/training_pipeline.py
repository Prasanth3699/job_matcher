"""
Training Pipeline for neural ranking models.

This module handles training, validation, and deployment of neural ranking models
with support for incremental learning and A/B testing.
"""

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    RankingConfig,
    FeedbackConfig,
    PerformanceMetrics,
    EnsembleConfig
)
from app.core.ranking.neural_ranker import NeuralRanker, LambdaRankNet, RankingFeatures
from app.core.ranking.feedback_collector import FeedbackCollector, FeedbackEvent


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_save_path: str = "./model_artifacts"
    checkpoint_frequency: int = 5
    use_gpu: bool = True


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    validation_loss: float
    ndcg_at_5: float
    ndcg_at_10: float
    map_score: float
    epoch: int
    timestamp: datetime


class RankingDataset(Dataset):
    """
    PyTorch dataset for ranking model training.
    
    Handles query-document pairs with relevance labels for learning-to-rank.
    """
    
    def __init__(
        self,
        features: List[np.ndarray],
        labels: List[float],
        query_ids: List[str],
        group_sizes: List[int]
    ):
        """
        Initialize ranking dataset.
        
        Args:
            features: List of feature vectors
            labels: List of relevance labels
            query_ids: List of query identifiers
            group_sizes: List of group sizes for each query
        """
        self.features = features
        self.labels = labels
        self.query_ids = query_ids
        self.group_sizes = group_sizes
        
        # Validate data consistency
        assert len(features) == len(labels) == len(query_ids)
        assert sum(group_sizes) == len(features)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        feature_tensor = torch.FloatTensor(self.features[idx])
        label_tensor = torch.FloatTensor([self.labels[idx]])
        return feature_tensor, label_tensor, self.query_ids[idx]
    
    def get_query_groups(self) -> List[Tuple[int, int]]:
        """Get start and end indices for each query group."""
        groups = []
        start_idx = 0
        
        for group_size in self.group_sizes:
            end_idx = start_idx + group_size
            groups.append((start_idx, end_idx))
            start_idx = end_idx
        
        return groups


class LambdaRankLoss(nn.Module):
    """
    LambdaRank loss function for learning-to-rank.
    
    Implements the LambdaRank algorithm that optimizes NDCG directly
    through gradient computation with pairwise comparisons.
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize LambdaRank loss.
        
        Args:
            sigma: Smoothing parameter for sigmoid function
        """
        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
    
    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        group_boundaries: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compute LambdaRank loss.
        
        Args:
            predictions: Model predictions
            labels: True relevance labels
            group_boundaries: Start and end indices for each query group
            
        Returns:
            LambdaRank loss value
        """
        total_loss = 0.0
        
        for start_idx, end_idx in group_boundaries:
            if end_idx - start_idx < 2:
                continue  # Need at least 2 documents for pairwise comparison
            
            group_predictions = predictions[start_idx:end_idx]
            group_labels = labels[start_idx:end_idx]
            
            # Compute pairwise loss for this group
            group_loss = self._compute_group_loss(group_predictions, group_labels)
            total_loss += group_loss
        
        return total_loss / len(group_boundaries) if group_boundaries else torch.tensor(0.0)
    
    def _compute_group_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for a single query group."""
        n_docs = len(predictions)
        loss = 0.0
        
        # Compute NDCG for current ranking
        current_ndcg = self._compute_ndcg(predictions, labels)
        
        # Pairwise comparisons
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                # Skip if labels are equal
                if labels[i] == labels[j]:
                    continue
                
                # Determine which document should be ranked higher
                if labels[i] > labels[j]:
                    # Document i should be ranked higher than j
                    s_ij = 1.0
                    higher_idx, lower_idx = i, j
                else:
                    # Document j should be ranked higher than i
                    s_ij = -1.0
                    higher_idx, lower_idx = j, i
                
                # Compute gain if we swap these documents
                delta_ndcg = self._compute_swap_delta_ndcg(
                    predictions, labels, higher_idx, lower_idx
                )
                
                # LambdaRank gradient weight
                lambda_weight = abs(delta_ndcg)
                
                # Pairwise loss with lambda weighting
                pred_diff = predictions[i] - predictions[j]
                sigmoid_term = torch.sigmoid(self.sigma * pred_diff)
                
                if s_ij > 0:  # i should be ranked higher
                    pairwise_loss = -torch.log(sigmoid_term + 1e-10)
                else:  # j should be ranked higher
                    pairwise_loss = -torch.log(1 - sigmoid_term + 1e-10)
                
                loss += lambda_weight * pairwise_loss
        
        return loss
    
    def _compute_ndcg(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        k: int = 10
    ) -> float:
        """Compute NDCG@k for current predictions."""
        try:
            # Convert to numpy for sklearn
            pred_np = predictions.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Reshape for sklearn (expects 2D)
            pred_2d = pred_np.reshape(1, -1)
            labels_2d = labels_np.reshape(1, -1)
            
            return ndcg_score(labels_2d, pred_2d, k=min(k, len(pred_np)))
        except:
            return 0.0
    
    def _compute_swap_delta_ndcg(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        idx1: int,
        idx2: int
    ) -> float:
        """Compute change in NDCG if we swap two documents."""
        try:
            # Original NDCG
            original_ndcg = self._compute_ndcg(predictions, labels)
            
            # Swap predictions
            swapped_predictions = predictions.clone()
            swapped_predictions[idx1], swapped_predictions[idx2] = \
                swapped_predictions[idx2], swapped_predictions[idx1]
            
            # New NDCG
            new_ndcg = self._compute_ndcg(swapped_predictions, labels)
            
            return new_ndcg - original_ndcg
        except:
            return 0.0


class TrainingPipeline:
    """
    Comprehensive training pipeline for neural ranking models with support
    for incremental learning, validation, and model deployment.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the training pipeline."""
        self.config = config or TrainingConfig()
        self.training_params = RankingConfig.TRAINING_PARAMS
        self.model_architecture = RankingConfig.MODEL_ARCHITECTURE
        
        # Training components
        self.neural_ranker: Optional[NeuralRanker] = None
        self.feedback_collector: Optional[FeedbackCollector] = None
        
        # Training state
        self.current_model: Optional[LambdaRankNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.loss_function = LambdaRankLoss()
        
        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.best_model_metrics: Optional[TrainingMetrics] = None
        self.early_stopping_counter = 0
        
        # Device configuration
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu'
        )
        
        logger.info(f"TrainingPipeline initialized on device: {self.device}")
    
    async def initialize(self):
        """Initialize async components."""
        try:
            self.neural_ranker = NeuralRanker()
            await self.neural_ranker.initialize()
            
            self.feedback_collector = FeedbackCollector()
            await self.feedback_collector.initialize()
            
            logger.info("TrainingPipeline async initialization completed")
        except Exception as e:
            logger.error(f"TrainingPipeline initialization failed: {str(e)}")
    
    async def prepare_training_data(
        self,
        min_feedback_count: int = 100,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Tuple[RankingDataset, RankingDataset]:
        """
        Prepare training and validation datasets from feedback data.
        
        Args:
            min_feedback_count: Minimum feedback events required
            time_range: Optional time range filter
            
        Returns:
            Tuple of (training_dataset, validation_dataset)
        """
        try:
            # Get training data from feedback collector
            training_data = await self.feedback_collector.get_training_data(
                min_feedback_count, time_range
            )
            
            if len(training_data) < min_feedback_count:
                raise ValueError(f"Insufficient training data: {len(training_data)} < {min_feedback_count}")
            
            # Process training data into features and labels
            features, labels, query_ids, group_sizes = await self._process_training_data(training_data)
            
            # Split into training and validation
            train_features, val_features, train_labels, val_labels, \
            train_query_ids, val_query_ids, train_groups, val_groups = \
                self._split_training_data(
                    features, labels, query_ids, group_sizes,
                    test_size=self.config.validation_split
                )
            
            # Create datasets
            train_dataset = RankingDataset(train_features, train_labels, train_query_ids, train_groups)
            val_dataset = RankingDataset(val_features, val_labels, val_query_ids, val_groups)
            
            logger.info(f"Training data prepared: {len(train_dataset)} train, {len(val_dataset)} validation")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise
    
    async def _process_training_data(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[float], List[str], List[int]]:
        """Process raw training data into features and labels."""
        try:
            features = []
            labels = []
            query_ids = []
            query_groups = {}
            
            for data_point in training_data:
                # Extract features using neural ranker
                resume_data = data_point.get('resume_data', {})
                job_data = data_point.get('job_data', {})
                
                # Extract ranking features
                ranking_features = await self.neural_ranker.extract_ranking_features(
                    resume_data, [job_data]
                )
                
                if not ranking_features:
                    continue
                
                feature_vector = ranking_features[0].to_tensor().numpy()
                
                # Convert feedback to relevance label
                feedback_events = data_point.get('feedback_events', [])
                relevance_label = self._compute_relevance_label(feedback_events)
                
                # Query ID (typically user_id or session_id)
                query_id = data_point.get('query_id', data_point.get('user_id', 'unknown'))
                
                features.append(feature_vector)
                labels.append(relevance_label)
                query_ids.append(query_id)
                
                # Group by query
                if query_id not in query_groups:
                    query_groups[query_id] = 0
                query_groups[query_id] += 1
            
            # Extract group sizes in order
            unique_queries = []
            group_sizes = []
            
            current_query = None
            for query_id in query_ids:
                if query_id != current_query:
                    if current_query is not None:
                        group_sizes.append(query_groups[current_query])
                    unique_queries.append(query_id)
                    current_query = query_id
            
            # Add the last group
            if current_query is not None:
                group_sizes.append(query_groups[current_query])
            
            logger.info(f"Processed {len(features)} training examples across {len(unique_queries)} queries")
            
            return features, labels, query_ids, group_sizes
            
        except Exception as e:
            logger.error(f"Training data processing failed: {str(e)}")
            raise
    
    def _compute_relevance_label(self, feedback_events: List[Dict[str, Any]]) -> float:
        """Compute relevance label from feedback events."""
        try:
            if not feedback_events:
                return 0.0
            
            # Weight feedback events
            total_weight = 0.0
            weighted_score = 0.0
            
            feedback_weights = FeedbackConfig.FEEDBACK_WEIGHTS
            
            for event in feedback_events:
                feedback_type = event.get('feedback_type', '')
                weight = feedback_weights.get(feedback_type, 0.0)
                
                # Apply temporal decay
                event_time = datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat()))
                days_ago = (datetime.now() - event_time).days
                decay_factor = FeedbackConfig.LEARNING_PARAMS['feedback_decay'] ** days_ago
                
                final_weight = weight * decay_factor
                
                if final_weight > 0:
                    weighted_score += final_weight
                    total_weight += abs(final_weight)
                elif final_weight < 0:
                    weighted_score += final_weight  # Negative feedback
                    total_weight += abs(final_weight)
            
            # Normalize to 0-1 range
            if total_weight > 0:
                normalized_score = max(0.0, min(1.0, (weighted_score + total_weight) / (2 * total_weight)))
            else:
                normalized_score = 0.5  # Neutral score for no feedback
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Relevance label computation failed: {str(e)}")
            return 0.0
    
    def _split_training_data(
        self,
        features: List[np.ndarray],
        labels: List[float],
        query_ids: List[str],
        group_sizes: List[int],
        test_size: float = 0.2
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], 
               List[str], List[str], List[int], List[int]]:
        """Split data maintaining query group integrity."""
        try:
            # Group queries for splitting
            unique_queries = []
            query_data = {}
            
            start_idx = 0
            for i, group_size in enumerate(group_sizes):
                end_idx = start_idx + group_size
                query_id = query_ids[start_idx]  # All items in group have same query_id
                
                if query_id not in unique_queries:
                    unique_queries.append(query_id)
                    query_data[query_id] = {
                        'features': features[start_idx:end_idx],
                        'labels': labels[start_idx:end_idx],
                        'query_ids': query_ids[start_idx:end_idx],
                        'group_size': group_size
                    }
                
                start_idx = end_idx
            
            # Split queries
            train_queries, val_queries = train_test_split(
                unique_queries, test_size=test_size, random_state=42
            )
            
            # Reconstruct training and validation sets
            train_features, train_labels, train_query_ids, train_groups = [], [], [], []
            val_features, val_labels, val_query_ids, val_groups = [], [], [], []
            
            for query_id in train_queries:
                data = query_data[query_id]
                train_features.extend(data['features'])
                train_labels.extend(data['labels'])
                train_query_ids.extend(data['query_ids'])
                train_groups.append(data['group_size'])
            
            for query_id in val_queries:
                data = query_data[query_id]
                val_features.extend(data['features'])
                val_labels.extend(data['labels'])
                val_query_ids.extend(data['query_ids'])
                val_groups.append(data['group_size'])
            
            return (train_features, val_features, train_labels, val_labels,
                    train_query_ids, val_query_ids, train_groups, val_groups)
            
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise
    
    async def train_model(
        self,
        train_dataset: RankingDataset,
        val_dataset: RankingDataset,
        model_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the neural ranking model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_save_path: Path to save the trained model
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting neural ranking model training")
            
            # Initialize model
            feature_dim = len(train_dataset[0][0])
            self.current_model = LambdaRankNet(feature_dim, self.model_architecture)
            self.current_model.to(self.device)
            
            # Initialize optimizer and scheduler
            self.optimizer = optim.AdamW(
                self.current_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            # Training loop
            best_val_loss = float('inf')
            self.early_stopping_counter = 0
            
            for epoch in range(self.config.epochs):
                # Training phase
                train_loss = await self._train_epoch(train_loader, train_dataset)
                
                # Validation phase
                val_loss, val_metrics = await self._validate_epoch(val_loader, val_dataset)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Create training metrics
                metrics = TrainingMetrics(
                    train_loss=train_loss,
                    validation_loss=val_loss,
                    ndcg_at_5=val_metrics.get('ndcg_at_5', 0.0),
                    ndcg_at_10=val_metrics.get('ndcg_at_10', 0.0),
                    map_score=val_metrics.get('map_score', 0.0),
                    epoch=epoch,
                    timestamp=datetime.now()
                )
                
                self.training_history.append(metrics)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_metrics = metrics
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    if model_save_path:
                        await self._save_model(model_save_path, metrics)
                else:
                    self.early_stopping_counter += 1
                
                # Log progress
                if epoch % 10 == 0 or self.early_stopping_counter == 0:
                    logger.info(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, ndcg@10={val_metrics.get('ndcg_at_10', 0):.4f}"
                    )
                
                # Early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Checkpoint saving
                if epoch % self.config.checkpoint_frequency == 0:
                    checkpoint_path = f"{model_save_path}_checkpoint_{epoch}.pt" if model_save_path else None
                    if checkpoint_path:
                        await self._save_checkpoint(checkpoint_path, epoch, metrics)
            
            # Training completed
            final_metrics = {
                'training_completed': True,
                'total_epochs': len(self.training_history),
                'best_metrics': self.best_model_metrics.__dict__ if self.best_model_metrics else {},
                'final_metrics': metrics.__dict__,
                'model_parameters': sum(p.numel() for p in self.current_model.parameters()),
                'device': str(self.device)
            }
            
            logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    async def _train_epoch(
        self,
        train_loader: DataLoader,
        train_dataset: RankingDataset
    ) -> float:
        """Train for one epoch."""
        self.current_model.train()
        total_loss = 0.0
        batch_count = 0
        
        group_boundaries = train_dataset.get_query_groups()
        
        for batch_idx, (features, labels, query_ids) in enumerate(train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.current_model(features)
            predictions = outputs['relevance'].squeeze()
            
            # Compute loss (simplified - would need proper group handling for batches)
            loss = nn.MSELoss()(predictions, labels.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
    async def _validate_epoch(
        self,
        val_loader: DataLoader,
        val_dataset: RankingDataset
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.current_model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        batch_count = 0
        
        with torch.no_grad():
            for features, labels, query_ids in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.current_model(features)
                predictions = outputs['relevance'].squeeze()
                
                # Compute loss
                loss = nn.MSELoss()(predictions, labels.squeeze())
                total_loss += loss.item()
                batch_count += 1
                
                # Store for metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        # Compute validation metrics
        metrics = self._compute_ranking_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def _compute_ranking_metrics(
        self,
        predictions: List[float],
        labels: List[float]
    ) -> Dict[str, float]:
        """Compute ranking evaluation metrics."""
        try:
            # Convert to numpy arrays
            pred_array = np.array(predictions).reshape(1, -1)
            label_array = np.array(labels).reshape(1, -1)
            
            # Compute NDCG scores
            ndcg_at_5 = ndcg_score(label_array, pred_array, k=5)
            ndcg_at_10 = ndcg_score(label_array, pred_array, k=10)
            
            # Simplified MAP computation
            sorted_indices = np.argsort(predictions)[::-1]
            sorted_labels = np.array(labels)[sorted_indices]
            
            relevant_items = np.where(sorted_labels > 0.5)[0]
            if len(relevant_items) > 0:
                precisions = []
                for i, relevant_idx in enumerate(relevant_items):
                    precision_at_k = (i + 1) / (relevant_idx + 1)
                    precisions.append(precision_at_k)
                map_score = np.mean(precisions)
            else:
                map_score = 0.0
            
            return {
                'ndcg_at_5': ndcg_at_5,
                'ndcg_at_10': ndcg_at_10,
                'map_score': map_score
            }
            
        except Exception as e:
            logger.error(f"Metrics computation failed: {str(e)}")
            return {'ndcg_at_5': 0.0, 'ndcg_at_10': 0.0, 'map_score': 0.0}
    
    async def _save_model(self, save_path: str, metrics: TrainingMetrics):
        """Save the trained model."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            model_state = {
                'model_state_dict': self.current_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics.__dict__,
                'config': self.config.__dict__,
                'model_architecture': self.model_architecture,
                'feature_dim': self.current_model.input_dim
            }
            
            torch.save(model_state, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
    
    async def _save_checkpoint(self, checkpoint_path: str, epoch: int, metrics: TrainingMetrics):
        """Save training checkpoint."""
        try:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': self.current_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics.__dict__,
                'training_history': [m.__dict__ for m in self.training_history]
            }
            
            torch.save(checkpoint_state, checkpoint_path)
            logger.debug(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {str(e)}")
    
    async def load_model(self, model_path: str) -> bool:
        """Load a trained model."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct model
            feature_dim = checkpoint['feature_dim']
            model_architecture = checkpoint['model_architecture']
            
            self.current_model = LambdaRankNet(feature_dim, model_architecture)
            self.current_model.load_state_dict(checkpoint['model_state_dict'])
            self.current_model.to(self.device)
            
            # Load metrics
            if 'metrics' in checkpoint:
                metrics_dict = checkpoint['metrics']
                self.best_model_metrics = TrainingMetrics(**metrics_dict)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False
    
    async def incremental_training(
        self,
        new_feedback_data: List[Dict[str, Any]],
        learning_rate: float = 0.0001,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Perform incremental training with new feedback data.
        
        Args:
            new_feedback_data: New feedback events for training
            learning_rate: Learning rate for incremental training
            epochs: Number of training epochs
            
        Returns:
            Incremental training results
        """
        try:
            if not self.current_model:
                raise ValueError("No model loaded for incremental training")
            
            logger.info(f"Starting incremental training with {len(new_feedback_data)} new samples")
            
            # Process new data
            features, labels, query_ids, group_sizes = await self._process_training_data(new_feedback_data)
            
            if not features:
                logger.warning("No valid features extracted from new feedback data")
                return {'success': False, 'reason': 'No valid features'}
            
            # Create dataset
            new_dataset = RankingDataset(features, labels, query_ids, group_sizes)
            new_loader = DataLoader(new_dataset, batch_size=16, shuffle=True)
            
            # Reduce learning rate for incremental training
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            # Incremental training loop
            initial_loss = None
            final_loss = None
            
            for epoch in range(epochs):
                epoch_loss = await self._train_epoch(new_loader, new_dataset)
                
                if epoch == 0:
                    initial_loss = epoch_loss
                final_loss = epoch_loss
                
                if epoch % 5 == 0:
                    logger.debug(f"Incremental epoch {epoch}: loss={epoch_loss:.4f}")
            
            # Compute improvement
            improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0
            
            results = {
                'success': True,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'improvement': improvement,
                'epochs_trained': epochs,
                'samples_processed': len(features)
            }
            
            logger.info(f"Incremental training completed. Loss improvement: {improvement:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Incremental training failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def evaluate_model(
        self,
        test_dataset: RankingDataset
    ) -> Dict[str, Any]:
        """Evaluate model on test dataset."""
        try:
            if not self.current_model:
                raise ValueError("No model loaded for evaluation")
            
            self.current_model.eval()
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            all_predictions = []
            all_labels = []
            total_loss = 0.0
            batch_count = 0
            
            with torch.no_grad():
                for features, labels, query_ids in test_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.current_model(features)
                    predictions = outputs['relevance'].squeeze()
                    
                    loss = nn.MSELoss()(predictions, labels.squeeze())
                    total_loss += loss.item()
                    batch_count += 1
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute comprehensive metrics
            ranking_metrics = self._compute_ranking_metrics(all_predictions, all_labels)
            
            evaluation_results = {
                'test_loss': total_loss / batch_count if batch_count > 0 else 0.0,
                'test_samples': len(test_dataset),
                'ranking_metrics': ranking_metrics,
                'model_info': await self.neural_ranker.get_model_info() if self.neural_ranker else {}
            }
            
            logger.info(f"Model evaluation completed on {len(test_dataset)} samples")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        try:
            return {
                'training_history': [m.__dict__ for m in self.training_history],
                'best_metrics': self.best_model_metrics.__dict__ if self.best_model_metrics else {},
                'training_config': self.config.__dict__,
                'model_architecture': self.model_architecture,
                'device': str(self.device),
                'total_epochs_trained': len(self.training_history),
                'early_stopping_counter': self.early_stopping_counter
            }
        except Exception as e:
            logger.error(f"Failed to get training stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Clean up training pipeline resources."""
        try:
            # Clean up neural ranker
            if self.neural_ranker:
                await self.neural_ranker.cleanup()
            
            # Clean up feedback collector
            if self.feedback_collector:
                await self.feedback_collector.cleanup()
            
            # Clear model from memory
            if self.current_model:
                del self.current_model
                self.current_model = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("TrainingPipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"TrainingPipeline cleanup failed: {str(e)}")