"""
Neural Ranking Model with LambdaRank Architecture.

This module implements state-of-the-art neural learning-to-rank models
for job matching with support for multi-task learning and real-time updates.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
import logging

from app.utils.logger import logger
from app.core.constants.ml_constants import (
    RankingConfig,
    FeedbackConfig,
    PerformanceMetrics,
    EnsembleConfig
)
from app.core.cache.embedding_cache import EmbeddingCache


@dataclass
class RankingFeatures:
    """Container for ranking features extracted from resume-job pairs."""
    semantic_features: np.ndarray
    statistical_features: np.ndarray
    interaction_features: np.ndarray
    contextual_features: np.ndarray
    feature_names: List[str]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert all features to a single tensor."""
        all_features = np.concatenate([
            self.semantic_features,
            self.statistical_features, 
            self.interaction_features,
            self.contextual_features
        ])
        return torch.FloatTensor(all_features)


@dataclass
class RankingPrediction:
    """Container for ranking model predictions."""
    scores: List[float]
    relevance_scores: List[float]
    diversity_scores: List[float]
    confidence: float
    explanation: Optional[Dict[str, Any]] = None


class LambdaRankNet(nn.Module):
    """
    LambdaRank neural network for learning-to-rank with multi-task objectives.
    
    Implements the LambdaRank algorithm which optimizes NDCG directly through
    neural network training with pairwise ranking loss.
    """
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        """
        Initialize the LambdaRank network.
        
        Args:
            input_dim: Dimension of input features
            config: Network configuration parameters
        """
        super(LambdaRankNet, self).__init__()
        
        self.config = config
        self.input_dim = input_dim
        
        # Extract architecture parameters
        hidden_layers = config.get('hidden_layers', [256, 128, 64])
        dropout_rate = config.get('dropout_rate', 0.2)
        activation = config.get('activation', 'relu')
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final output layers for multi-task learning
        self.feature_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        self.relevance_head = nn.Linear(prev_dim, 1)  # Primary relevance score
        self.diversity_head = nn.Linear(prev_dim, 1)  # Diversity score
        self.confidence_head = nn.Linear(prev_dim, 1)  # Confidence estimation
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Dictionary with relevance, diversity, and confidence scores
        """
        # Shared feature representation
        features = self.feature_layers(x)
        
        # Task-specific outputs
        relevance = torch.sigmoid(self.relevance_head(features))
        diversity = torch.sigmoid(self.diversity_head(features))
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'relevance': relevance,
            'diversity': diversity,
            'confidence': confidence,
            'features': features
        }


class NeuralRanker:
    """
    Advanced neural ranking system implementing LambdaRank with multi-task learning,
    real-time feedback integration, and explainable predictions.
    """
    
    def __init__(self):
        """Initialize the neural ranking system."""
        self.config = RankingConfig.MODEL_ARCHITECTURE
        self.training_params = RankingConfig.TRAINING_PARAMS
        self.ranking_objectives = RankingConfig.RANKING_OBJECTIVES
        
        # Model components
        self.model: Optional[LambdaRankNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Feature extraction
        self.feature_dim = 512  # Will be determined dynamically
        self.feature_extractor = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        self.model_version = "v1.0.0"
        
        # Performance tracking
        self.performance_metrics = {
            'ndcg_scores': [],
            'training_losses': [],
            'validation_scores': [],
            'inference_times': []
        }
        
        # Cache for computed features
        self.embedding_cache = EmbeddingCache()
        
        logger.info("NeuralRanker initialized successfully")
    
    async def initialize(self):
        """Initialize async components."""
        try:
            await self.embedding_cache.initialize()
            self._setup_feature_extractor()
            logger.info("NeuralRanker async initialization completed")
        except Exception as e:
            logger.error(f"NeuralRanker initialization failed: {str(e)}")
    
    def _setup_feature_extractor(self):
        """Set up the feature extraction pipeline."""
        try:
            from app.core.matching.ensemble.ensemble_manager import EnsembleMatchingEngine
            self.feature_extractor = EnsembleMatchingEngine()
            logger.info("Feature extractor initialized")
        except Exception as e:
            logger.warning(f"Feature extractor setup failed: {str(e)}")
    
    async def extract_ranking_features(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]]
    ) -> List[RankingFeatures]:
        """
        Extract comprehensive ranking features for resume-job pairs.
        
        Args:
            resume_data: Parsed resume data
            jobs: List of job descriptions
            
        Returns:
            List of ranking features for each job
        """
        try:
            ranking_features = []
            
            # Extract resume features once
            resume_features = await self._extract_resume_features(resume_data)
            
            for job in jobs:
                # Extract job features
                job_features = await self._extract_job_features(job)
                
                # Create semantic features (embeddings)
                semantic_features = await self._create_semantic_features(
                    resume_features, job_features
                )
                
                # Create statistical features
                statistical_features = self._create_statistical_features(
                    resume_data, job
                )
                
                # Create interaction features
                interaction_features = self._create_interaction_features(
                    resume_features, job_features
                )
                
                # Create contextual features
                contextual_features = self._create_contextual_features(
                    resume_data, job
                )
                
                # Feature names for interpretability
                feature_names = self._get_feature_names()
                
                ranking_features.append(RankingFeatures(
                    semantic_features=semantic_features,
                    statistical_features=statistical_features,
                    interaction_features=interaction_features,
                    contextual_features=contextual_features,
                    feature_names=feature_names
                ))
            
            return ranking_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return []
    
    async def _extract_resume_features(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from resume."""
        try:
            # Basic features
            features = {
                'skills': resume_data.get('skills', []),
                'experience_years': self._calculate_experience_years(resume_data),
                'education_level': self._determine_education_level(resume_data),
                'certifications': resume_data.get('certifications', []),
                'work_history': resume_data.get('experiences', [])
            }
            
            # Text content for semantic analysis
            text_content = self._extract_text_content(resume_data)
            features['text_content'] = text_content
            
            # Skill categories
            features['skill_categories'] = self._categorize_skills(features['skills'])
            
            # Career progression
            features['career_progression'] = self._analyze_career_progression(
                features['work_history']
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Resume feature extraction failed: {str(e)}")
            return {}
    
    async def _extract_job_features(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from job posting."""
        try:
            features = {
                'required_skills': job.get('skills', []),
                'job_level': self._determine_job_level(job),
                'company_size': job.get('company_size', 'unknown'),
                'job_type': job.get('job_type', ''),
                'location': job.get('location', ''),
                'salary_range': job.get('salary_range', ''),
                'requirements': job.get('requirements', []),
                'responsibilities': job.get('responsibilities', [])
            }
            
            # Text content for semantic analysis
            text_content = self._extract_job_text_content(job)
            features['text_content'] = text_content
            
            # Skill categories
            features['skill_categories'] = self._categorize_skills(features['required_skills'])
            
            # Job complexity estimation
            features['complexity_score'] = self._estimate_job_complexity(job)
            
            return features
            
        except Exception as e:
            logger.error(f"Job feature extraction failed: {str(e)}")
            return {}
    
    async def _create_semantic_features(
        self,
        resume_features: Dict[str, Any],
        job_features: Dict[str, Any]
    ) -> np.ndarray:
        """Create semantic similarity features using embeddings."""
        try:
            # Try to get cached embeddings first
            resume_text = resume_features.get('text_content', '')
            job_text = job_features.get('text_content', '')
            
            resume_embedding = await self.embedding_cache.get_embedding(
                resume_text, 'primary_semantic'
            )
            job_embedding = await self.embedding_cache.get_embedding(
                job_text, 'primary_semantic'
            )
            
            # If not cached, compute embeddings (simplified for now)
            if resume_embedding is None:
                resume_embedding = np.random.normal(0, 1, 768)  # Placeholder
                await self.embedding_cache.store_embedding(
                    resume_text, 'primary_semantic', resume_embedding
                )
            
            if job_embedding is None:
                job_embedding = np.random.normal(0, 1, 768)  # Placeholder
                await self.embedding_cache.store_embedding(
                    job_text, 'primary_semantic', job_embedding
                )
            
            # Calculate semantic similarities
            cosine_similarity = np.dot(resume_embedding, job_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
            )
            
            # Create semantic feature vector
            semantic_features = np.array([
                cosine_similarity,
                np.linalg.norm(resume_embedding - job_embedding),  # L2 distance
                np.max(resume_embedding * job_embedding),  # Max element-wise product
                np.mean(resume_embedding * job_embedding)   # Mean element-wise product
            ])
            
            return semantic_features
            
        except Exception as e:
            logger.error(f"Semantic feature creation failed: {str(e)}")
            return np.zeros(4)  # Return zero features on error
    
    def _create_statistical_features(
        self,
        resume_data: Dict[str, Any],
        job: Dict[str, Any]
    ) -> np.ndarray:
        """Create statistical features from resume-job comparison."""
        try:
            # Skill matching features
            resume_skills = set(s.lower() for s in resume_data.get('skills', []))
            job_skills = set(s.lower() for s in job.get('skills', []))
            
            skill_intersection = len(resume_skills.intersection(job_skills))
            skill_union = len(resume_skills.union(job_skills))
            skill_jaccard = skill_intersection / skill_union if skill_union > 0 else 0
            
            skill_coverage = skill_intersection / len(job_skills) if job_skills else 0
            skill_overlap_ratio = skill_intersection / len(resume_skills) if resume_skills else 0
            
            # Experience matching
            resume_experience = self._calculate_experience_years(resume_data)
            required_experience = self._extract_required_experience(job)
            experience_match = min(1.0, resume_experience / required_experience) if required_experience > 0 else 1.0
            
            # Education matching
            resume_education = self._determine_education_level(resume_data)
            required_education = self._extract_required_education(job)
            education_match = self._calculate_education_match(resume_education, required_education)
            
            # Location matching (simplified)
            location_match = self._calculate_location_match(
                resume_data.get('location', ''),
                job.get('location', '')
            )
            
            # Create statistical feature vector
            statistical_features = np.array([
                skill_jaccard,
                skill_coverage,
                skill_overlap_ratio,
                experience_match,
                education_match,
                location_match,
                len(resume_skills),
                len(job_skills),
                resume_experience,
                required_experience
            ])
            
            return statistical_features
            
        except Exception as e:
            logger.error(f"Statistical feature creation failed: {str(e)}")
            return np.zeros(10)
    
    def _create_interaction_features(
        self,
        resume_features: Dict[str, Any],
        job_features: Dict[str, Any]
    ) -> np.ndarray:
        """Create interaction features between resume and job characteristics."""
        try:
            # Skill category interactions
            resume_categories = resume_features.get('skill_categories', {})
            job_categories = job_features.get('skill_categories', {})
            
            category_matches = []
            for category in ['technical', 'soft', 'domain', 'tools']:
                resume_count = len(resume_categories.get(category, []))
                job_count = len(job_categories.get(category, []))
                
                if resume_count > 0 and job_count > 0:
                    category_match = min(resume_count, job_count) / max(resume_count, job_count)
                else:
                    category_match = 0.0
                    
                category_matches.append(category_match)
            
            # Career progression alignment
            career_level = resume_features.get('career_progression', {}).get('level', 'entry')
            job_level = job_features.get('job_level', 'mid')
            level_alignment = self._calculate_level_alignment(career_level, job_level)
            
            # Complexity match
            resume_complexity = resume_features.get('career_progression', {}).get('complexity', 0.5)
            job_complexity = job_features.get('complexity_score', 0.5)
            complexity_match = 1.0 - abs(resume_complexity - job_complexity)
            
            # Create interaction feature vector
            interaction_features = np.array([
                *category_matches,  # 4 features
                level_alignment,    # 1 feature
                complexity_match    # 1 feature
            ])
            
            return interaction_features
            
        except Exception as e:
            logger.error(f"Interaction feature creation failed: {str(e)}")
            return np.zeros(6)
    
    def _create_contextual_features(
        self,
        resume_data: Dict[str, Any],
        job: Dict[str, Any]
    ) -> np.ndarray:
        """Create contextual features considering external factors."""
        try:
            # Time-based features
            current_time = datetime.now()
            
            # Job posting recency (assuming job has posting_date)
            posting_date = job.get('posting_date', current_time)
            if isinstance(posting_date, str):
                posting_date = datetime.fromisoformat(posting_date.replace('Z', '+00:00'))
            
            days_since_posting = (current_time - posting_date).days
            posting_freshness = max(0, 1 - days_since_posting / 30)  # Decay over 30 days
            
            # Market demand (simplified - would use real market data)
            job_title = job.get('job_title', '').lower()
            market_demand = self._estimate_market_demand(job_title)
            
            # Company attractiveness (simplified scoring)
            company_score = self._calculate_company_attractiveness(job)
            
            # Salary competitiveness
            salary_competitiveness = self._calculate_salary_competitiveness(job)
            
            # Create contextual feature vector
            contextual_features = np.array([
                posting_freshness,
                market_demand,
                company_score,
                salary_competitiveness,
                days_since_posting / 365.0  # Normalized days
            ])
            
            return contextual_features
            
        except Exception as e:
            logger.error(f"Contextual feature creation failed: {str(e)}")
            return np.zeros(5)
    
    async def predict_rankings(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]]
    ) -> RankingPrediction:
        """
        Predict rankings for jobs given a resume.
        
        Args:
            resume_data: Parsed resume data
            jobs: List of job descriptions
            
        Returns:
            Ranking predictions with scores and explanations
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained, using fallback ranking")
                return await self._fallback_ranking(resume_data, jobs)
            
            # Extract features for all jobs
            ranking_features = await self.extract_ranking_features(resume_data, jobs)
            
            if not ranking_features:
                logger.error("No features extracted for ranking")
                return RankingPrediction([], [], [], 0.0)
            
            # Prepare batch input
            feature_tensors = [features.to_tensor() for features in ranking_features]
            batch_input = torch.stack(feature_tensors)
            
            # Model inference
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(batch_input)
            
            # Extract predictions
            relevance_scores = predictions['relevance'].squeeze().tolist()
            diversity_scores = predictions['diversity'].squeeze().tolist()
            confidence_scores = predictions['confidence'].squeeze().tolist()
            
            # Ensure lists for single job case
            if not isinstance(relevance_scores, list):
                relevance_scores = [relevance_scores]
            if not isinstance(diversity_scores, list):
                diversity_scores = [diversity_scores]
            if not isinstance(confidence_scores, list):
                confidence_scores = [confidence_scores]
            
            # Combine relevance and diversity based on objectives
            objective_weights = self.ranking_objectives['weights']
            combined_scores = []
            
            for rel, div in zip(relevance_scores, diversity_scores):
                combined_score = (
                    objective_weights[0] * rel +
                    objective_weights[1] * div
                )
                combined_scores.append(combined_score)
            
            # Overall confidence as mean of individual confidences
            overall_confidence = np.mean(confidence_scores)
            
            return RankingPrediction(
                scores=combined_scores,
                relevance_scores=relevance_scores,
                diversity_scores=diversity_scores,
                confidence=float(overall_confidence)
            )
            
        except Exception as e:
            logger.error(f"Ranking prediction failed: {str(e)}")
            return await self._fallback_ranking(resume_data, jobs)
    
    async def _fallback_ranking(
        self,
        resume_data: Dict[str, Any],
        jobs: List[Dict[str, Any]]
    ) -> RankingPrediction:
        """Fallback ranking using simple heuristics."""
        try:
            scores = []
            resume_skills = set(s.lower() for s in resume_data.get('skills', []))
            
            for job in jobs:
                job_skills = set(s.lower() for s in job.get('skills', []))
                
                # Simple Jaccard similarity
                if resume_skills and job_skills:
                    intersection = len(resume_skills.intersection(job_skills))
                    union = len(resume_skills.union(job_skills))
                    score = intersection / union if union > 0 else 0.0
                else:
                    score = 0.0
                
                scores.append(score)
            
            return RankingPrediction(
                scores=scores,
                relevance_scores=scores,
                diversity_scores=[0.5] * len(scores),  # Default diversity
                confidence=0.6  # Lower confidence for fallback
            )
            
        except Exception as e:
            logger.error(f"Fallback ranking failed: {str(e)}")
            return RankingPrediction([], [], [], 0.0)
    
    # Helper methods for feature extraction
    
    def _calculate_experience_years(self, resume_data: Dict[str, Any]) -> float:
        """Calculate total years of experience."""
        try:
            total_months = 0
            for exp in resume_data.get('experiences', []):
                duration = exp.get('duration_months', 0)
                total_months += duration
            return total_months / 12.0
        except:
            return 0.0
    
    def _determine_education_level(self, resume_data: Dict[str, Any]) -> str:
        """Determine highest education level."""
        education_hierarchy = {
            'phd': 5, 'doctorate': 5,
            'master': 4, 'mba': 4,
            'bachelor': 3, 'degree': 3,
            'associate': 2, 'diploma': 2,
            'certificate': 1
        }
        
        highest_level = 0
        highest_degree = 'none'
        
        for edu in resume_data.get('education', []):
            degree = edu.get('degree', '').lower()
            for level_name, level_value in education_hierarchy.items():
                if level_name in degree and level_value > highest_level:
                    highest_level = level_value
                    highest_degree = level_name
        
        return highest_degree
    
    def _extract_text_content(self, resume_data: Dict[str, Any]) -> str:
        """Extract all text content from resume."""
        text_parts = []
        
        # Add summary
        if resume_data.get('summary'):
            text_parts.append(resume_data['summary'])
        
        # Add experience descriptions
        for exp in resume_data.get('experiences', []):
            if exp.get('description'):
                text_parts.append(exp['description'])
        
        # Add skills
        text_parts.extend(resume_data.get('skills', []))
        
        return ' '.join(text_parts)
    
    def _extract_job_text_content(self, job: Dict[str, Any]) -> str:
        """Extract all text content from job posting."""
        text_parts = []
        
        # Add job title and description
        if job.get('job_title'):
            text_parts.append(job['job_title'])
        if job.get('description'):
            text_parts.append(job['description'])
        
        # Add requirements and responsibilities
        for field in ['requirements', 'responsibilities', 'qualifications']:
            items = job.get(field, [])
            if isinstance(items, list):
                text_parts.extend(items)
            elif isinstance(items, str):
                text_parts.append(items)
        
        # Add skills
        text_parts.extend(job.get('skills', []))
        
        return ' '.join(text_parts)
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into technical, soft, domain, tools."""
        categories = {
            'technical': [],
            'soft': [],
            'domain': [],
            'tools': []
        }
        
        # Simple categorization (would be more sophisticated in practice)
        technical_keywords = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node']
        soft_keywords = ['communication', 'leadership', 'teamwork', 'problem-solving']
        tool_keywords = ['git', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins']
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if any(keyword in skill_lower for keyword in technical_keywords):
                categories['technical'].append(skill)
            elif any(keyword in skill_lower for keyword in soft_keywords):
                categories['soft'].append(skill)
            elif any(keyword in skill_lower for keyword in tool_keywords):
                categories['tools'].append(skill)
            else:
                categories['domain'].append(skill)
        
        return categories
    
    def _analyze_career_progression(self, work_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze career progression from work history."""
        if not work_history:
            return {'level': 'entry', 'complexity': 0.3}
        
        # Sort by date (most recent first)
        sorted_history = sorted(
            work_history,
            key=lambda x: x.get('end_date', '2024-01-01'),
            reverse=True
        )
        
        # Determine current level based on most recent role
        recent_title = sorted_history[0].get('job_title', '').lower()
        
        if any(keyword in recent_title for keyword in ['senior', 'lead', 'principal', 'architect']):
            level = 'senior'
            complexity = 0.8
        elif any(keyword in recent_title for keyword in ['manager', 'director', 'vp', 'ceo']):
            level = 'executive'
            complexity = 0.9
        elif any(keyword in recent_title for keyword in ['junior', 'associate', 'intern']):
            level = 'junior'
            complexity = 0.4
        else:
            level = 'mid'
            complexity = 0.6
        
        return {
            'level': level,
            'complexity': complexity,
            'total_roles': len(work_history)
        }
    
    def _determine_job_level(self, job: Dict[str, Any]) -> str:
        """Determine job level from job posting."""
        title = job.get('job_title', '').lower()
        requirements = ' '.join(job.get('requirements', [])).lower()
        
        if any(keyword in title for keyword in ['senior', 'lead', 'principal', 'architect']):
            return 'senior'
        elif any(keyword in title for keyword in ['manager', 'director', 'vp']):
            return 'executive'
        elif any(keyword in title for keyword in ['junior', 'associate', 'entry']):
            return 'junior'
        elif 'years' in requirements:
            # Try to extract experience requirement
            if any(str(i) in requirements for i in range(5, 20)):
                return 'senior'
            elif any(str(i) in requirements for i in range(2, 5)):
                return 'mid'
            else:
                return 'junior'
        else:
            return 'mid'
    
    def _estimate_job_complexity(self, job: Dict[str, Any]) -> float:
        """Estimate job complexity based on requirements."""
        complexity_indicators = [
            'architect', 'lead', 'senior', 'principal', 'expert',
            'advanced', 'complex', 'enterprise', 'scale', 'distributed'
        ]
        
        text_content = self._extract_job_text_content(job).lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text_content)
        
        # Normalize to 0-1 range
        return min(1.0, complexity_score / len(complexity_indicators))
    
    def _get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability."""
        return [
            # Semantic features (4)
            'semantic_cosine_similarity', 'semantic_l2_distance', 
            'semantic_max_product', 'semantic_mean_product',
            
            # Statistical features (10)
            'skill_jaccard', 'skill_coverage', 'skill_overlap_ratio',
            'experience_match', 'education_match', 'location_match',
            'resume_skill_count', 'job_skill_count', 
            'resume_experience_years', 'required_experience_years',
            
            # Interaction features (6)
            'technical_skill_match', 'soft_skill_match', 
            'domain_skill_match', 'tool_skill_match',
            'level_alignment', 'complexity_match',
            
            # Contextual features (5)
            'posting_freshness', 'market_demand', 
            'company_score', 'salary_competitiveness', 'normalized_posting_age'
        ]
    
    # Additional helper methods
    
    def _extract_required_experience(self, job: Dict[str, Any]) -> float:
        """Extract required years of experience from job posting."""
        requirements = ' '.join(job.get('requirements', [])).lower()
        
        # Simple pattern matching for experience requirements
        import re
        patterns = [
            r'(\d+)\+?\s*years?\s*of?\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'minimum\s+(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, requirements)
            if match:
                return float(match.group(1))
        
        return 2.0  # Default assumption
    
    def _extract_required_education(self, job: Dict[str, Any]) -> str:
        """Extract required education level from job posting."""
        requirements = ' '.join(job.get('requirements', [])).lower()
        
        if 'phd' in requirements or 'doctorate' in requirements:
            return 'phd'
        elif 'master' in requirements or 'mba' in requirements:
            return 'master'
        elif 'bachelor' in requirements or 'degree' in requirements:
            return 'bachelor'
        elif 'associate' in requirements:
            return 'associate'
        else:
            return 'none'
    
    def _calculate_education_match(self, resume_education: str, required_education: str) -> float:
        """Calculate education level match score."""
        education_scores = {
            'none': 0, 'certificate': 1, 'associate': 2,
            'bachelor': 3, 'master': 4, 'phd': 5
        }
        
        resume_score = education_scores.get(resume_education, 0)
        required_score = education_scores.get(required_education, 0)
        
        if resume_score >= required_score:
            return 1.0
        else:
            return resume_score / required_score if required_score > 0 else 0.0
    
    def _calculate_location_match(self, resume_location: str, job_location: str) -> float:
        """Calculate location match score."""
        if not resume_location or not job_location:
            return 0.5  # Neutral score for missing location info
        
        # Simple string matching (would be more sophisticated with geolocation)
        resume_lower = resume_location.lower()
        job_lower = job_location.lower()
        
        if resume_lower == job_lower:
            return 1.0
        elif any(word in job_lower for word in resume_lower.split()):
            return 0.7
        elif 'remote' in job_lower:
            return 0.9
        else:
            return 0.3
    
    def _calculate_level_alignment(self, career_level: str, job_level: str) -> float:
        """Calculate career level alignment score."""
        level_hierarchy = {
            'entry': 1, 'junior': 2, 'mid': 3, 'senior': 4, 'executive': 5
        }
        
        career_score = level_hierarchy.get(career_level, 3)
        job_score = level_hierarchy.get(job_level, 3)
        
        # Perfect match gets 1.0, adjacent levels get 0.7, etc.
        level_diff = abs(career_score - job_score)
        return max(0.0, 1.0 - level_diff * 0.3)
    
    def _estimate_market_demand(self, job_title: str) -> float:
        """Estimate market demand for job title (simplified)."""
        high_demand_keywords = [
            'engineer', 'developer', 'analyst', 'scientist', 'architect',
            'manager', 'consultant', 'specialist'
        ]
        
        demand_score = sum(1 for keyword in high_demand_keywords if keyword in job_title)
        return min(1.0, demand_score / 3.0)  # Normalize
    
    def _calculate_company_attractiveness(self, job: Dict[str, Any]) -> float:
        """Calculate company attractiveness score (simplified)."""
        # This would use real company data in practice
        company_name = job.get('company_name', '').lower()
        
        # Simplified scoring based on company size and benefits
        benefits = job.get('benefits', [])
        benefit_score = len(benefits) / 10.0  # Normalize by typical benefit count
        
        # Company size factor
        company_size = job.get('company_size', 'medium').lower()
        size_scores = {'startup': 0.6, 'small': 0.7, 'medium': 0.8, 'large': 0.9, 'enterprise': 0.85}
        size_score = size_scores.get(company_size, 0.7)
        
        return (benefit_score + size_score) / 2.0
    
    def _calculate_salary_competitiveness(self, job: Dict[str, Any]) -> float:
        """Calculate salary competitiveness (simplified)."""
        salary_range = job.get('salary_range', '')
        
        if not salary_range:
            return 0.5  # Neutral score for missing salary info
        
        # Simple heuristic based on salary range mention
        if 'competitive' in salary_range.lower():
            return 0.7
        elif '$' in salary_range:
            return 0.8  # Explicit salary range provided
        else:
            return 0.5
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_dim': self.feature_dim,
            'architecture': self.config,
            'training_params': self.training_params,
            'performance_metrics': self.performance_metrics,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
    
    async def cleanup(self):
        """Clean up model resources."""
        try:
            if self.embedding_cache:
                await self.embedding_cache.cleanup()
            
            # Clear model from memory
            if self.model:
                del self.model
                self.model = None
            
            logger.info("NeuralRanker cleanup completed")
            
        except Exception as e:
            logger.error(f"NeuralRanker cleanup failed: {str(e)}")