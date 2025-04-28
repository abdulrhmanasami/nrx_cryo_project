"""
NRX-Cryo Industrial Performance Monitor v2.2
============================================
Mission-Critical System Resource Monitoring
- Enhanced CPU/GPU Accuracy
- Multi-GPU Support
- Industrial-Grade Tag Management
"""

import time
import os
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
    )
except ImportError:
    pass

# Custom Exceptions
class PerformanceMonitorError(Exception):
    """Base exception for monitoring failures"""

class GPUMonitoringError(PerformanceMonitorError):
    """Exception for GPU-related monitoring issues"""

@dataclass
class PerformanceSnapshot:
    """Industrial-grade monitoring data container"""
    timestamp: float
    cpu_usage: float
    ram_usage: float
    gpu_usage: Optional[List[float]] = None  # List for multi-GPU support
    gpu_memory: Optional[List[float]] = None

class PerformanceMonitor:
    """Industrial System Resource Monitor with Multi-GPU Support"""
    
    def __init__(self, enable_gpu: bool = False, precision: str = 'ms'):
        """
        Initialize performance monitor with enterprise-grade configuration
        
        Args:
            enable_gpu: Enable NVIDIA GPU monitoring (requires pynvml)
            precision: Timing precision (ms = milliseconds, us = microseconds)
        """
        self.enable_gpu = enable_gpu
        self.precision = 1e3 if precision == 'ms' else 1e6
        self._gpu_available = False
        self._gpu_handles: List[Any] = []
        self._sessions: List[Dict] = []
        self._current_session: Dict[str, Dict] = {}
        self._enable_monitoring = os.getenv('PERFORMANCE_MONITORING', '1') == '1'

        if self.enable_gpu:
            self._init_gpu_monitoring()

    def _init_gpu_monitoring(self) -> None:
        """Initialize NVIDIA GPU monitoring subsystem with multi-GPU support"""
        try:
            nvmlInit()
            gpu_count = nvmlDeviceGetCount()
            if gpu_count > 0:
                self._gpu_available = True
                self._gpu_handles = [
                    nvmlDeviceGetHandleByIndex(i) 
                    for i in range(gpu_count)
                ]
        except Exception as e:
            raise GPUMonitoringError(f"Failed to initialize GPU monitoring: {str(e)}")
        finally:
            if not self._gpu_available and self.enable_gpu:
                nvmlShutdown()

    def start_monitoring(self, tag: str = 'default') -> None:
        """Start resource tracking session with nested capability and tag collision check"""
        if not self._enable_monitoring:
            return

        if tag in self._current_session:
            raise PerformanceMonitorError(f"Tag '{tag}' is already in use!")

        # Prime CPU measurement for accuracy
        psutil.Process().cpu_percent(interval=0.1)
        
        snapshot = self._take_snapshot()
        self._current_session[tag] = {
            'start_time': time.perf_counter(),
            'start_snapshot': snapshot,
            'children': []
        }

    def stop_monitoring(self, tag: str = 'default') -> Dict[str, Any]:
        """Finalize monitoring session and calculate industrial-grade metrics"""
        if not self._enable_monitoring:
            return {}

        snapshot = self._take_snapshot()
        session = self._current_session.pop(tag, None)
        
        if not session:
            raise PerformanceMonitorError(f"No active session found for tag: {tag}")

        metrics = self._calculate_metrics(session['start_snapshot'], snapshot)
        metrics['execution_time'] = (time.perf_counter() - session['start_time']) * self.precision
        
        if self._sessions:
            self._sessions[-1]['children'].append(metrics)
        else:
            self._sessions.append(metrics)
            
        return metrics

    def _take_snapshot(self) -> PerformanceSnapshot:
        """Capture system resource state with industrial precision"""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage=process.cpu_percent(interval=0.1),  # Accurate CPU measurement
            ram_usage=mem_info.rss / 1024**2  # Convert to MB
        )

        if self._gpu_available:
            try:
                snapshot.gpu_usage = []
                snapshot.gpu_memory = []
                for handle in self._gpu_handles:
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    snapshot.gpu_usage.append(mem_info.used / mem_info.total * 100)
                    snapshot.gpu_memory.append(mem_info.used / 1024**2)
            except Exception as e:
                raise GPUMonitoringError(f"GPU snapshot failed: {str(e)}")

        return snapshot

    def _calculate_metrics(self, start: PerformanceSnapshot, end: PerformanceSnapshot) -> Dict[str, Any]:
        """Calculate performance deltas with industrial validation"""
        metrics = {
            'cpu_delta': end.cpu_usage - start.cpu_usage,
            'ram_delta': end.ram_usage - start.ram_usage,
            'ram_peak': max(start.ram_usage, end.ram_usage),
        }

        if self._gpu_available:
            metrics.update({
                'gpu_delta': [e - s for e, s in zip(end.gpu_memory, start.gpu_memory)],
                'gpu_peak': [
                    max(start_gpu, end_gpu) 
                    for start_gpu, end_gpu in zip(start.gpu_memory, end.gpu_memory)
                ]
            })
        
        return metrics

    def report(self, format: str = 'dict') -> Any:
        """Generate industrial QA report in multiple formats"""
        if not self._enable_monitoring:
            return {} if format == 'dict' else ''

        base_report = self._sessions[-1] if self._sessions else {}
        
        if format == 'dict':
            return base_report
        elif format == 'json':
            import json
            return json.dumps(base_report, indent=2)
        elif format == 'text':
            return self._format_text_report(base_report)
        else:
            raise PerformanceMonitorError(f"Unsupported report format: {format}")

    def _format_text_report(self, data: Dict) -> str:
        """Generate human-readable industrial report"""
        time_unit = 'ms' if self.precision == 1e3 else 'us'
        report = [
            "NRX-Cryo Performance Report",
            "===========================",
            f"Execution Time: {data['execution_time']:.2f}{time_unit}",
            f"Memory Usage: +{data['ram_delta']:.2f}MB (Peak: {data['ram_peak']:.2f}MB)"
        ]
        
        if self._gpu_available and data.get('gpu_delta'):
            for gpu_id, (delta, peak) in enumerate(zip(data['gpu_delta'], data['gpu_peak'])):
                report.append(
                    f"GPU {gpu_id} Memory: +{delta:.2f}MB (Peak: {peak:.2f}MB)"
                )
            
        if data.get('children'):
            report.append("\nNested Operations:")
            for child in data['children']:
                report.append(
                    f"  - {child['execution_time']:.2f}{time_unit} "
                    f"(+{child['ram_delta']:.2f}MB)"
                )
                
        return '\n'.join(report)

    def log_performance(self, logger, level: str = 'INFO') -> None:
        """Integrated logging with NRX-Cryo LoggerSystem and level validation"""
        if not self._enable_monitoring:
            return

        if not hasattr(logger, level.lower()):
            raise PerformanceMonitorError(f"Logger doesn't support level: {level}")

        report = self.report('text')
        log_method = getattr(logger, level.lower())
        log_method(f"\n{report}")

    def reset(self) -> None:
        """Reset all monitoring sessions for batch processing"""
        self._sessions.clear()
        self._current_session.clear()

    def __enter__(self):
        """Context manager entry point"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        self.stop_monitoring()

# Example Usage
if __name__ == "__main__":
    from logger_system import LoggerSystem  # Assume external logger exists
    
    # Initialize monitoring system
    monitor = PerformanceMonitor(enable_gpu=True, precision='ms')
    logger_sys = LoggerSystem()
    logger = logger_sys.get_logger()

    # Monitor main process
    with monitor:
        # Monitor nested operation
        monitor.start_monitoring('data_processing')
        time.sleep(0.1)
        monitor.stop_monitoring('data_processing')
        
        # Monitor another nested operation
        monitor.start_monitoring('rendering')
        time.sleep(0.2)
        monitor.stop_monitoring('rendering')

    # Generate reports
    print(monitor.report('text'))
    monitor.log_performance(logger, 'INFO')