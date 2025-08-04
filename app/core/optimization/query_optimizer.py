"""
Query Optimizer for database and search query optimization.

This module provides intelligent query optimization, execution planning,
and performance monitoring for database and search operations.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json

from app.utils.logger import logger


class QueryType(str, Enum):
    """Types of queries that can be optimized."""
    DATABASE = "database"
    SEARCH = "search"
    AGGREGATION = "aggregation"
    VECTOR = "vector"
    HYBRID = "hybrid"


class OptimizationStrategy(str, Enum):
    """Query optimization strategies."""
    INDEX_BASED = "index_based"
    CACHING = "caching"
    BATCHING = "batching"
    PARALLEL = "parallel"
    MATERIALIZED_VIEW = "materialized_view"
    QUERY_REWRITE = "query_rewrite"


@dataclass
class QueryPlan:
    """Query execution plan."""
    query_id: str
    query_type: QueryType
    original_query: Dict[str, Any]
    optimized_query: Dict[str, Any]
    strategies: List[OptimizationStrategy]
    estimated_cost: float
    estimated_time_ms: float
    index_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_id': self.query_id,
            'query_type': self.query_type.value,
            'original_query': self.original_query,
            'optimized_query': self.optimized_query,
            'strategies': [s.value for s in self.strategies],
            'estimated_cost': self.estimated_cost,
            'estimated_time_ms': self.estimated_time_ms,
            'index_recommendations': self.index_recommendations
        }


@dataclass
class QueryExecution:
    """Query execution results and metrics."""
    query_id: str
    query_plan: QueryPlan
    actual_time_ms: float
    rows_processed: int
    cache_hit: bool
    error: Optional[str]
    execution_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_id': self.query_id,
            'query_plan': self.query_plan.to_dict(),
            'actual_time_ms': self.actual_time_ms,
            'rows_processed': self.rows_processed,
            'cache_hit': self.cache_hit,
            'error': self.error,
            'execution_timestamp': self.execution_timestamp.isoformat()
        }


@dataclass
class QueryStats:
    """Query performance statistics."""
    total_queries: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    cache_hit_ratio: float
    optimization_hit_ratio: float
    error_rate: float
    queries_by_type: Dict[str, int]
    top_slow_queries: List[str]


class QueryOptimizer:
    """
    Advanced query optimizer that analyzes, optimizes, and monitors
    database and search query performance.
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        enable_batching: bool = True,
        cache_ttl_seconds: int = 300,
        batch_size: int = 100
    ):
        """Initialize query optimizer."""
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.batch_size = batch_size
        
        # Query tracking
        self.query_history: Dict[str, List[QueryExecution]] = {}
        self.query_plans: Dict[str, QueryPlan] = {}
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Batch processing
        self.pending_batches: Dict[str, List[Dict[str, Any]]] = {}
        self.batch_processors: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.stats = QueryStats(
            total_queries=0,
            total_execution_time_ms=0.0,
            average_execution_time_ms=0.0,
            cache_hit_ratio=0.0,
            optimization_hit_ratio=0.0,
            error_rate=0.0,
            queries_by_type={},
            top_slow_queries=[]
        )
        
        # Optimization rules
        self.optimization_rules = {
            QueryType.DATABASE: self._optimize_database_query,
            QueryType.SEARCH: self._optimize_search_query,
            QueryType.AGGREGATION: self._optimize_aggregation_query,
            QueryType.VECTOR: self._optimize_vector_query,
            QueryType.HYBRID: self._optimize_hybrid_query
        }
        
        # Index recommendations
        self.index_recommendations: Dict[str, List[str]] = {}
        self.query_patterns: Dict[str, int] = {}
        
        logger.info("QueryOptimizer initialized")
    
    async def optimize_query(
        self,
        query: Dict[str, Any],
        query_type: QueryType,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Optimize a query and return execution plan.
        
        Args:
            query: Query to optimize
            query_type: Type of query
            context: Additional context for optimization
            
        Returns:
            Optimized query execution plan
        """
        try:
            query_id = self._generate_query_id(query, query_type)
            
            # Check if we have a cached plan
            if query_id in self.query_plans:
                cached_plan = self.query_plans[query_id]
                logger.debug(f"Using cached query plan for {query_id}")
                return cached_plan
            
            # Analyze query
            analysis = await self._analyze_query(query, query_type, context)
            
            # Apply optimization strategies
            optimized_query = query.copy()
            strategies = []
            
            # Apply type-specific optimizations
            if query_type in self.optimization_rules:
                optimizer_result = await self.optimization_rules[query_type](
                    optimized_query, analysis, context
                )
                optimized_query = optimizer_result['query']
                strategies.extend(optimizer_result['strategies'])
            
            # Apply general optimizations
            general_optimizations = await self._apply_general_optimizations(
                optimized_query, analysis, context
            )
            optimized_query = general_optimizations['query']
            strategies.extend(general_optimizations['strategies'])
            
            # Estimate cost and time
            cost_estimate = await self._estimate_query_cost(optimized_query, query_type)
            time_estimate = await self._estimate_execution_time(optimized_query, query_type)
            
            # Generate index recommendations
            index_recommendations = await self._generate_index_recommendations(
                optimized_query, query_type, analysis
            )
            
            # Create query plan
            query_plan = QueryPlan(
                query_id=query_id,
                query_type=query_type,
                original_query=query,
                optimized_query=optimized_query,
                strategies=strategies,
                estimated_cost=cost_estimate,
                estimated_time_ms=time_estimate,
                index_recommendations=index_recommendations
            )
            
            # Cache the plan
            self.query_plans[query_id] = query_plan
            
            logger.debug(f"Optimized query {query_id} with strategies: {[s.value for s in strategies]}")
            
            return query_plan
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            
            # Return basic plan as fallback
            return QueryPlan(
                query_id=self._generate_query_id(query, query_type),
                query_type=query_type,
                original_query=query,
                optimized_query=query,
                strategies=[],
                estimated_cost=1.0,
                estimated_time_ms=100.0,
                index_recommendations=[]
            )
    
    async def execute_query(
        self,
        query_plan: QueryPlan,
        executor_function: Callable,
        *args,
        **kwargs
    ) -> QueryExecution:
        """
        Execute optimized query and track performance.
        
        Args:
            query_plan: Optimized query plan
            executor_function: Function to execute the query
            *args, **kwargs: Arguments for executor function
            
        Returns:
            Query execution results
        """
        try:
            start_time = time.time()
            cache_hit = False
            error = None
            rows_processed = 0
            result = None
            
            # Check cache if caching is enabled
            if (self.enable_caching and 
                OptimizationStrategy.CACHING in query_plan.strategies):
                
                cached_result = await self._check_query_cache(query_plan.query_id)
                if cached_result is not None:
                    result = cached_result
                    cache_hit = True
                    logger.debug(f"Cache hit for query {query_plan.query_id}")
            
            # Execute query if not cached
            if not cache_hit:
                try:
                    if asyncio.iscoroutinefunction(executor_function):
                        result = await executor_function(
                            query_plan.optimized_query, *args, **kwargs
                        )
                    else:
                        result = executor_function(
                            query_plan.optimized_query, *args, **kwargs
                        )
                    
                    # Cache result if caching is enabled
                    if (self.enable_caching and 
                        OptimizationStrategy.CACHING in query_plan.strategies):
                        await self._cache_query_result(query_plan.query_id, result)
                    
                    # Extract rows processed (if available)
                    if isinstance(result, dict) and 'rows_processed' in result:
                        rows_processed = result['rows_processed']
                    elif isinstance(result, list):
                        rows_processed = len(result)
                    
                except Exception as exec_error:
                    error = str(exec_error)
                    logger.error(f"Query execution failed: {error}")
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create execution record
            execution = QueryExecution(
                query_id=query_plan.query_id,
                query_plan=query_plan,
                actual_time_ms=execution_time_ms,
                rows_processed=rows_processed,
                cache_hit=cache_hit,
                error=error,
                execution_timestamp=datetime.now()
            )
            
            # Track execution
            await self._track_query_execution(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Query execution tracking failed: {str(e)}")
            
            # Return error execution
            return QueryExecution(
                query_id=query_plan.query_id,
                query_plan=query_plan,
                actual_time_ms=0.0,
                rows_processed=0,
                cache_hit=False,
                error=str(e),
                execution_timestamp=datetime.now()
            )
    
    async def _analyze_query(
        self,
        query: Dict[str, Any],
        query_type: QueryType,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze query structure and requirements."""
        try:
            analysis = {
                'complexity_score': 1.0,
                'table_scans': [],
                'join_operations': [],
                'filter_conditions': [],
                'aggregations': [],
                'sort_operations': [],
                'limit_clause': None,
                'estimated_selectivity': 1.0
            }
            
            # Database query analysis
            if query_type == QueryType.DATABASE:
                if 'tables' in query:
                    analysis['table_scans'] = query['tables']
                if 'joins' in query:
                    analysis['join_operations'] = query['joins']
                    analysis['complexity_score'] += len(query['joins']) * 0.5
                if 'where' in query:
                    analysis['filter_conditions'] = query['where']
                    analysis['estimated_selectivity'] = 0.1  # Assume filters are selective
                if 'group_by' in query:
                    analysis['aggregations'].append('group_by')
                    analysis['complexity_score'] += 0.3
                if 'order_by' in query:
                    analysis['sort_operations'] = query['order_by']
                    analysis['complexity_score'] += 0.2
                if 'limit' in query:
                    analysis['limit_clause'] = query['limit']
            
            # Search query analysis
            elif query_type == QueryType.SEARCH:
                if 'query_string' in query:
                    # Analyze search complexity
                    query_string = query['query_string']
                    word_count = len(query_string.split())
                    analysis['complexity_score'] = min(word_count * 0.1, 2.0)
                
                if 'filters' in query:
                    analysis['filter_conditions'] = query['filters']
                    analysis['estimated_selectivity'] = 0.2
            
            # Vector query analysis
            elif query_type == QueryType.VECTOR:
                if 'vector_dimension' in query:
                    dim = query['vector_dimension']
                    analysis['complexity_score'] = min(dim / 1000, 3.0)
                
                if 'similarity_threshold' in query:
                    threshold = query['similarity_threshold']
                    analysis['estimated_selectivity'] = 1.0 - threshold
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return {'complexity_score': 1.0}
    
    async def _optimize_database_query(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize database query."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Add indexes for filter conditions
            if analysis.get('filter_conditions'):
                strategies.append(OptimizationStrategy.INDEX_BASED)
            
            # Optimize joins
            if analysis.get('join_operations'):
                # Reorder joins by selectivity
                if len(analysis['join_operations']) > 1:
                    strategies.append(OptimizationStrategy.QUERY_REWRITE)
                    # Implement join reordering logic here
            
            # Add limit if not present for large result sets
            if not analysis.get('limit_clause') and analysis.get('complexity_score', 0) > 2.0:
                optimized_query['limit'] = 1000
                strategies.append(OptimizationStrategy.QUERY_REWRITE)
            
            # Consider caching for complex queries
            if analysis.get('complexity_score', 0) > 1.5:
                strategies.append(OptimizationStrategy.CACHING)
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"Database query optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _optimize_search_query(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize search query."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Add filters for better performance
            if not analysis.get('filter_conditions'):
                # Add default filters if context provides them
                if context and 'default_filters' in context:
                    optimized_query['filters'] = context['default_filters']
                    strategies.append(OptimizationStrategy.QUERY_REWRITE)
            
            # Consider caching for common search patterns
            if analysis.get('complexity_score', 0) < 1.0:
                strategies.append(OptimizationStrategy.CACHING)
            
            # Add result size limits
            if 'size' not in optimized_query:
                optimized_query['size'] = 50
                strategies.append(OptimizationStrategy.QUERY_REWRITE)
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"Search query optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _optimize_aggregation_query(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize aggregation query."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Consider materialized views for complex aggregations
            if analysis.get('complexity_score', 0) > 2.0:
                strategies.append(OptimizationStrategy.MATERIALIZED_VIEW)
            
            # Use caching for expensive aggregations
            strategies.append(OptimizationStrategy.CACHING)
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"Aggregation query optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _optimize_vector_query(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize vector similarity query."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Optimize vector search parameters
            if 'similarity_threshold' not in optimized_query:
                optimized_query['similarity_threshold'] = 0.7
                strategies.append(OptimizationStrategy.QUERY_REWRITE)
            
            # Consider approximate search for large vectors
            if analysis.get('complexity_score', 0) > 2.0:
                optimized_query['approximate'] = True
                strategies.append(OptimizationStrategy.QUERY_REWRITE)
            
            # Cache common vector queries
            strategies.append(OptimizationStrategy.CACHING)
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"Vector query optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _optimize_hybrid_query(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize hybrid query combining multiple sources."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Break down into parallel sub-queries
            if 'sub_queries' in query and len(query['sub_queries']) > 1:
                strategies.append(OptimizationStrategy.PARALLEL)
            
            # Consider batching for multiple similar queries
            if analysis.get('complexity_score', 0) > 1.0:
                strategies.append(OptimizationStrategy.BATCHING)
            
            # Cache hybrid query results
            strategies.append(OptimizationStrategy.CACHING)
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"Hybrid query optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _apply_general_optimizations(
        self,
        query: Dict[str, Any],
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply general optimization strategies."""
        try:
            optimized_query = query.copy()
            strategies = []
            
            # Add batching for eligible queries
            if (self.enable_batching and 
                analysis.get('complexity_score', 0) < 2.0 and
                context and context.get('batchable', False)):
                strategies.append(OptimizationStrategy.BATCHING)
            
            # Add caching for repetitive queries
            query_pattern = self._extract_query_pattern(query)
            if query_pattern in self.query_patterns:
                self.query_patterns[query_pattern] += 1
                if self.query_patterns[query_pattern] > 5:
                    strategies.append(OptimizationStrategy.CACHING)
            else:
                self.query_patterns[query_pattern] = 1
            
            return {
                'query': optimized_query,
                'strategies': strategies
            }
            
        except Exception as e:
            logger.error(f"General optimization failed: {str(e)}")
            return {'query': query, 'strategies': []}
    
    async def _estimate_query_cost(
        self,
        query: Dict[str, Any],
        query_type: QueryType
    ) -> float:
        """Estimate query execution cost."""
        try:
            base_cost = 1.0
            
            # Database queries
            if query_type == QueryType.DATABASE:
                # Cost based on table scans and joins
                table_count = len(query.get('tables', []))
                join_count = len(query.get('joins', []))
                
                base_cost = table_count * 0.5 + join_count * 1.0
                
                # Reduce cost if using indexes
                if query.get('where'):
                    base_cost *= 0.5
            
            # Search queries
            elif query_type == QueryType.SEARCH:
                query_length = len(query.get('query_string', ''))
                base_cost = min(query_length * 0.01, 2.0)
            
            # Vector queries
            elif query_type == QueryType.VECTOR:
                dimension = query.get('vector_dimension', 100)
                base_cost = dimension / 100
            
            return max(base_cost, 0.1)
            
        except Exception as e:
            logger.error(f"Cost estimation failed: {str(e)}")
            return 1.0
    
    async def _estimate_execution_time(
        self,
        query: Dict[str, Any],
        query_type: QueryType
    ) -> float:
        """Estimate query execution time in milliseconds."""
        try:
            # Base time estimates by query type
            base_times = {
                QueryType.DATABASE: 50.0,
                QueryType.SEARCH: 30.0,
                QueryType.AGGREGATION: 100.0,
                QueryType.VECTOR: 200.0,
                QueryType.HYBRID: 150.0
            }
            
            base_time = base_times.get(query_type, 100.0)
            
            # Adjust based on complexity
            complexity_factor = 1.0
            
            if query_type == QueryType.DATABASE:
                join_count = len(query.get('joins', []))
                complexity_factor = 1.0 + (join_count * 0.5)
            
            elif query_type == QueryType.VECTOR:
                dimension = query.get('vector_dimension', 100)
                complexity_factor = dimension / 100
            
            return base_time * complexity_factor
            
        except Exception as e:
            logger.error(f"Time estimation failed: {str(e)}")
            return 100.0
    
    async def _generate_index_recommendations(
        self,
        query: Dict[str, Any],
        query_type: QueryType,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate index recommendations for the query."""
        try:
            recommendations = []
            
            if query_type == QueryType.DATABASE:
                # Recommend indexes for filter conditions
                for condition in analysis.get('filter_conditions', []):
                    if isinstance(condition, dict) and 'column' in condition:
                        recommendations.append(f"INDEX ON {condition['column']}")
                
                # Recommend composite indexes for joins
                for join in analysis.get('join_operations', []):
                    if isinstance(join, dict):
                        join_columns = join.get('columns', [])
                        if len(join_columns) > 1:
                            recommendations.append(f"COMPOSITE INDEX ON {', '.join(join_columns)}")
            
            elif query_type == QueryType.SEARCH:
                # Recommend full-text indexes
                if 'query_string' in query:
                    recommendations.append("FULL_TEXT INDEX ON content")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Index recommendation generation failed: {str(e)}")
            return []
    
    def _generate_query_id(self, query: Dict[str, Any], query_type: QueryType) -> str:
        """Generate unique ID for query."""
        try:
            query_string = json.dumps(query, sort_keys=True)
            query_hash = hashlib.md5(f"{query_type.value}:{query_string}".encode()).hexdigest()
            return f"{query_type.value}_{query_hash[:12]}"
        except Exception:
            return f"{query_type.value}_{int(time.time())}"
    
    def _extract_query_pattern(self, query: Dict[str, Any]) -> str:
        """Extract query pattern for repetition detection."""
        try:
            # Create pattern by removing specific values
            pattern_query = {}
            
            for key, value in query.items():
                if key in ['limit', 'offset', 'page']:
                    pattern_query[key] = 'VARIABLE'
                elif isinstance(value, (str, int, float)):
                    pattern_query[key] = 'VALUE'
                else:
                    pattern_query[key] = value
            
            return json.dumps(pattern_query, sort_keys=True)
            
        except Exception:
            return str(hash(str(query)))
    
    async def _check_query_cache(self, query_id: str) -> Optional[Any]:
        """Check if query result is cached."""
        try:
            if query_id in self.query_cache:
                result, timestamp = self.query_cache[query_id]
                
                # Check if cache is still valid
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                    return result
                else:
                    # Remove expired cache entry
                    del self.query_cache[query_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Query cache check failed: {str(e)}")
            return None
    
    async def _cache_query_result(self, query_id: str, result: Any):
        """Cache query result."""
        try:
            self.query_cache[query_id] = (result, datetime.now())
            
            # Limit cache size
            if len(self.query_cache) > 1000:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.query_cache.items(),
                    key=lambda x: x[1][1]
                )
                
                for i in range(100):  # Remove 100 oldest entries
                    del self.query_cache[sorted_cache[i][0]]
            
        except Exception as e:
            logger.error(f"Query result caching failed: {str(e)}")
    
    async def _track_query_execution(self, execution: QueryExecution):
        """Track query execution for performance monitoring."""
        try:
            query_id = execution.query_id
            
            # Add to history
            if query_id not in self.query_history:
                self.query_history[query_id] = []
            
            self.query_history[query_id].append(execution)
            
            # Keep only recent executions
            if len(self.query_history[query_id]) > 100:
                self.query_history[query_id] = self.query_history[query_id][-100:]
            
            # Update stats
            self.stats.total_queries += 1
            self.stats.total_execution_time_ms += execution.actual_time_ms
            
            if self.stats.total_queries > 0:
                self.stats.average_execution_time_ms = (
                    self.stats.total_execution_time_ms / self.stats.total_queries
                )
            
            # Update query type stats
            query_type = execution.query_plan.query_type.value
            if query_type not in self.stats.queries_by_type:
                self.stats.queries_by_type[query_type] = 0
            self.stats.queries_by_type[query_type] += 1
            
            # Update top slow queries
            if execution.actual_time_ms > 1000:  # Queries slower than 1 second
                if query_id not in self.stats.top_slow_queries:
                    self.stats.top_slow_queries.append(query_id)
                    
                # Keep only top 10 slow queries
                if len(self.stats.top_slow_queries) > 10:
                    self.stats.top_slow_queries = self.stats.top_slow_queries[-10:]
            
            # Update cache hit ratio
            cache_hits = sum(
                1 for executions in self.query_history.values()
                for exec in executions if exec.cache_hit
            )
            
            if self.stats.total_queries > 0:
                self.stats.cache_hit_ratio = cache_hits / self.stats.total_queries
            
            # Update optimization hit ratio
            optimized_queries = sum(
                1 for executions in self.query_history.values()
                for exec in executions if exec.query_plan.strategies
            )
            
            if self.stats.total_queries > 0:
                self.stats.optimization_hit_ratio = optimized_queries / self.stats.total_queries
            
            # Update error rate
            error_queries = sum(
                1 for executions in self.query_history.values()
                for exec in executions if exec.error
            )
            
            if self.stats.total_queries > 0:
                self.stats.error_rate = error_queries / self.stats.total_queries
            
        except Exception as e:
            logger.error(f"Query execution tracking failed: {str(e)}")
    
    async def get_query_stats(self) -> Dict[str, Any]:
        """Get comprehensive query performance statistics."""
        try:
            return {
                'performance_stats': asdict(self.stats),
                'cache_info': {
                    'cached_queries': len(self.query_cache),
                    'cache_ttl_seconds': self.cache_ttl_seconds
                },
                'optimization_info': {
                    'cached_plans': len(self.query_plans),
                    'query_patterns': len(self.query_patterns),
                    'index_recommendations': len(self.index_recommendations)
                },
                'recent_executions': sum(len(history) for history in self.query_history.values()),
                'configuration': {
                    'caching_enabled': self.enable_caching,
                    'batching_enabled': self.enable_batching,
                    'batch_size': self.batch_size
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get query stats: {str(e)}")
            return {}
    
    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries for analysis."""
        try:
            all_executions = []
            
            for query_id, executions in self.query_history.items():
                for execution in executions:
                    all_executions.append(execution)
            
            # Sort by execution time
            all_executions.sort(key=lambda x: x.actual_time_ms, reverse=True)
            
            # Return top slow queries
            slow_queries = []
            for execution in all_executions[:limit]:
                slow_queries.append({
                    'query_id': execution.query_id,
                    'execution_time_ms': execution.actual_time_ms,
                    'query_type': execution.query_plan.query_type.value,
                    'original_query': execution.query_plan.original_query,
                    'optimization_strategies': [s.value for s in execution.query_plan.strategies],
                    'execution_timestamp': execution.execution_timestamp.isoformat()
                })
            
            return slow_queries
            
        except Exception as e:
            logger.error(f"Failed to get slow queries: {str(e)}")
            return []
    
    async def clear_cache(self):
        """Clear query cache and plans."""
        try:
            self.query_cache.clear()
            self.query_plans.clear()
            logger.info("Query cache and plans cleared")
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {str(e)}")
    
    async def cleanup(self):
        """Clean up query optimizer resources."""
        try:
            # Cancel batch processors
            for task in self.batch_processors.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clear data structures
            self.query_history.clear()
            self.query_plans.clear()
            self.query_cache.clear()
            self.pending_batches.clear()
            self.batch_processors.clear()
            self.index_recommendations.clear()
            self.query_patterns.clear()
            
            logger.info("QueryOptimizer cleanup completed")
            
        except Exception as e:
            logger.error(f"QueryOptimizer cleanup failed: {str(e)}")