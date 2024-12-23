from typing import Dict, Any, Optional, Callable
import traceback
import logging
import asyncio
from time_utils import TimeUtils
from dataclasses import dataclass
from enum import Enum
import json

class ErrorSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ErrorCategory(Enum):
    MEMORY = "memory"
    LEARNING = "learning"
    KNOWLEDGE = "knowledge"
    INTERACTION = "interaction"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_attempts: int = 0

class ErrorHandler:
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.max_recovery_attempts = 3
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'unrecovered_errors': 0
        }

    def register_recovery_strategy(self, 
                                 category: ErrorCategory,
                                 severity: ErrorSeverity,
                                 strategy: Callable) -> None:
        """Register a recovery strategy for a specific error category and severity"""
        key = (category, severity)
        self.recovery_strategies[key] = strategy

    async def handle_error(self, 
                          error: Exception,
                          category: ErrorCategory,
                          system_state: Dict[str, Any],
                          severity: ErrorSeverity = None) -> Optional[Dict[str, Any]]:
        """Handle an error with appropriate recovery strategy"""
        # Create error context
        error_context = ErrorContext(
            timestamp=TimeUtils.format_timestamp(TimeUtils.get_current_time()),
            severity=severity or self._assess_severity(error),
            category=category,
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
            system_state=system_state
        )

        # Log error
        self._log_error(error_context)
        
        # Update stats
        self.error_stats['total_errors'] += 1
        
        # Attempt recovery
        try:
            recovery_result = await self._attempt_recovery(error_context)
            if recovery_result:
                self.error_stats['recovered_errors'] += 1
                return recovery_result
        except Exception as recovery_error:
            logging.error(f"Recovery failed: {str(recovery_error)}")
            
        self.error_stats['unrecovered_errors'] += 1
        return None

    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess the severity of an error"""
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, Warning):
            return ErrorSeverity.LOW
        return ErrorSeverity.HIGH

    async def _attempt_recovery(self, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """Attempt to recover from an error"""
        if context.recovery_attempts >= self.max_recovery_attempts:
            logging.warning(f"Max recovery attempts reached for error: {context.message}")
            return None

        strategy = self.recovery_strategies.get((context.category, context.severity))
        if not strategy:
            strategy = self._get_default_recovery_strategy(context)

        if strategy:
            context.recovery_attempts += 1
            return await strategy(context)
        
        return None

    def _get_default_recovery_strategy(self, context: ErrorContext) -> Optional[Callable]:
        """Get a default recovery strategy based on error context"""
        if context.severity == ErrorSeverity.CRITICAL:
            return self._critical_error_recovery
        elif context.category == ErrorCategory.MEMORY:
            return self._memory_error_recovery
        elif context.category == ErrorCategory.LEARNING:
            return self._learning_error_recovery
        return None

    async def _critical_error_recovery(self, context: ErrorContext) -> Dict[str, Any]:
        """Default recovery strategy for critical errors"""
        logging.critical(f"Attempting critical error recovery: {context.message}")
        
        # Save current state
        self._save_emergency_backup(context)
        
        # Reset system state
        new_state = self._create_clean_state()
        
        return {
            'status': 'recovered',
            'action': 'system_reset',
            'new_state': new_state
        }

    async def _memory_error_recovery(self, context: ErrorContext) -> Dict[str, Any]:
        """Default recovery strategy for memory errors"""
        logging.error(f"Attempting memory error recovery: {context.message}")
        
        # Free up memory
        self._cleanup_memory(context.system_state)
        
        return {
            'status': 'recovered',
            'action': 'memory_cleanup',
            'system_state': context.system_state
        }

    async def _learning_error_recovery(self, context: ErrorContext) -> Dict[str, Any]:
        """Default recovery strategy for learning errors"""
        logging.error(f"Attempting learning error recovery: {context.message}")
        
        # Reset learning parameters
        context.system_state['learning_params'] = self._get_default_learning_params()
        
        return {
            'status': 'recovered',
            'action': 'reset_learning',
            'system_state': context.system_state
        }

    def _log_error(self, context: ErrorContext) -> None:
        """Log error details"""
        log_message = (
            f"Error: {context.error_type}\n"
            f"Severity: {context.severity.name}\n"
            f"Category: {context.category.value}\n"
            f"Message: {context.message}\n"
            f"Timestamp: {context.timestamp}\n"
            f"Stack Trace:\n{context.stack_trace}"
        )
        logging.error(log_message)
        self.error_history.append(context)

    def _save_emergency_backup(self, context: ErrorContext) -> None:
        """Save emergency backup of system state"""
        backup_file = f"emergency_backup_{TimeUtils.get_current_time().timestamp()}.json"
        with open(backup_file, 'w') as f:
            json.dump({
                'error_context': {
                    'timestamp': context.timestamp,
                    'severity': context.severity.name,
                    'category': context.category.value,
                    'error_type': context.error_type,
                    'message': context.message
                },
                'system_state': context.system_state
            }, f, indent=2)

    def _create_clean_state(self) -> Dict[str, Any]:
        """Create a clean system state"""
        return {
            'initialized': True,
            'timestamp': TimeUtils.format_timestamp(TimeUtils.get_current_time()),
            'memory': {},
            'learning_params': self._get_default_learning_params(),
            'error_recovery': True
        }

    def _cleanup_memory(self, system_state: Dict[str, Any]) -> None:
        """Clean up memory to recover from memory errors"""
        if 'memory' in system_state:
            # Remove old or less important memories
            if 'short_term' in system_state['memory']:
                system_state['memory']['short_term'] = []
            if 'working_memory' in system_state['memory']:
                system_state['memory']['working_memory'] = {}

    def _get_default_learning_params(self) -> Dict[str, float]:
        """Get default learning parameters"""
        return {
            'learning_rate': 0.1,
            'memory_threshold': 0.5,
            'importance_threshold': 0.3,
            'consolidation_threshold': 0.7
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            **self.error_stats,
            'error_history_length': len(self.error_history),
            'recovery_strategies': len(self.recovery_strategies),
            'last_error_timestamp': (
                self.error_history[-1].timestamp 
                if self.error_history else None
            )
        }
