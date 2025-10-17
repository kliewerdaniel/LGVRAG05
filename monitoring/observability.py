"""
Production monitoring and observability for RAG system.

This module implements comprehensive logging, metrics collection,
tracing, and alerting for production RAG deployments.
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    query_length: Optional[int] = None
    response_length: Optional[int] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    chunks_retrieved: Optional[int] = None
    tokens_used: Optional[int] = None


class MetricsCollector:
    """
    Collects and aggregates system and application metrics.

    This class provides real-time monitoring of system resources,
    request performance, and application health.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize metrics collector."""
        self.config = get_config(config_path)

        # Metrics storage
        self.request_metrics: deque = deque(maxlen=10000)
        self.system_snapshots: deque = deque(maxlen=1000)

        # Aggregation windows (in seconds)
        self.aggregation_windows = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }

        # Aggregated metrics storage
        self.aggregated_metrics = {window: {} for window in self.aggregation_windows}

        # Background collection threads
        self._stop_collection = threading.Event()
        self._collection_thread = None

        # Start background collection
        self.start_collection()

    def start_collection(self) -> None:
        """Start background metrics collection."""
        if self._collection_thread is None:
            self._collection_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
            self._collection_thread.start()
            logger.info("Started background metrics collection")

    def stop_collection(self) -> None:
        """Stop background metrics collection."""
        self._stop_collection.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
            logger.info("Stopped background metrics collection")

    def _collect_system_metrics(self) -> None:
        """Background system metrics collection."""
        while not self._stop_collection.is_set():
            try:
                snapshot = self._get_system_snapshot()
                self.system_snapshots.append(snapshot)

                # Aggregate metrics periodically
                self._aggregate_metrics()

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            # Collect every 10 seconds
            self._stop_collection.wait(10)

    def _get_system_snapshot(self) -> MetricsSnapshot:
        """Get current system metrics snapshot."""
        timestamp = time.time()

        # CPU and memory
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            }
        except ImportError:
            # psutil not available
            cpu_percent = 0.0
            memory_percent = 0.0
            disk_usage_percent = 0.0
            network_io = {"bytes_sent": 0, "bytes_recv": 0}

        # GPU metrics (if available)
        gpu_metrics = []
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                })
        except ImportError:
            # GPU monitoring not available
            pass

        return MetricsSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_io=network_io,
            gpu_metrics=gpu_metrics
        )

    def record_request(self, metrics: RequestMetrics) -> None:
        """Record metrics for a completed request."""
        # Calculate duration if not provided
        if metrics.end_time and not metrics.duration:
            metrics.duration = metrics.end_time - metrics.start_time

        self.request_metrics.append(metrics)

        # Log slow requests
        if metrics.duration and metrics.duration > 10.0:  # 10 second threshold
            logger.warning(f"Slow request detected: {metrics.duration:.2f}s - {metrics.endpoint}")

        # Log errors
        if metrics.error_message:
            logger.error(f"Request error: {metrics.error_message} - {metrics.endpoint}")

    def _aggregate_metrics(self) -> None:
        """Aggregate metrics over time windows."""
        current_time = time.time()

        for window_name, window_seconds in self.aggregation_windows.items():
            window_start = current_time - window_seconds

            # Filter requests in this window
            window_requests = [
                m for m in self.request_metrics
                if m.start_time >= window_start
            ]

            if window_requests:
                # Calculate aggregated statistics
                durations = [m.duration for m in window_requests if m.duration]
                status_codes = [m.status_code for m in window_requests if m.status_code]

                aggregated = {
                    "request_count": len(window_requests),
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "min_duration": min(durations) if durations else 0,
                    "max_duration": max(durations) if durations else 0,
                    "error_count": len([m for m in window_requests if m.error_message]),
                    "success_rate": len([c for c in status_codes if c < 400]) / len(status_codes) if status_codes else 0,
                    "window_start": window_start,
                    "window_end": current_time
                }

                self.aggregated_metrics[window_name] = aggregated

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics."""
        # Get latest system snapshot
        latest_snapshot = self.system_snapshots[-1] if self.system_snapshots else None

        # Get recent request statistics
        recent_requests = list(self.request_metrics)[-100:]  # Last 100 requests

        current_metrics = {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": latest_snapshot.cpu_percent if latest_snapshot else 0,
                "memory_percent": latest_snapshot.memory_percent if latest_snapshot else 0,
                "disk_usage_percent": latest_snapshot.disk_usage_percent if latest_snapshot else 0,
                "gpu_metrics": latest_snapshot.gpu_metrics if latest_snapshot else []
            },
            "requests": {
                "total_recent": len(recent_requests),
                "avg_duration": sum(m.duration for m in recent_requests if m.duration) / len([m for m in recent_requests if m.duration]) if recent_requests else 0,
                "error_rate": len([m for m in recent_requests if m.error_message]) / len(recent_requests) if recent_requests else 0
            },
            "aggregated": self.aggregated_metrics.copy()
        }

        return current_metrics

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        metrics = self.get_current_metrics()

        # Define health thresholds
        cpu_threshold = 80.0
        memory_threshold = 85.0
        error_rate_threshold = 0.05  # 5%

        health_issues = []

        # Check CPU
        if metrics["system"]["cpu_percent"] > cpu_threshold:
            health_issues.append(f"High CPU usage: {metrics['system']['cpu_percent']:.1f}%")

        # Check memory
        if metrics["system"]["memory_percent"] > memory_threshold:
            health_issues.append(f"High memory usage: {metrics['system']['memory_percent']:.1f}%")

        # Check error rate
        if metrics["requests"]["error_rate"] > error_rate_threshold:
            health_issues.append(f"High error rate: {metrics['requests']['error_rate']:.2%}")

        # Determine overall health
        if health_issues:
            status = "unhealthy"
            message = "; ".join(health_issues)
        else:
            status = "healthy"
            message = "All systems operational"

        return {
            "status": status,
            "message": message,
            "timestamp": metrics["timestamp"],
            "issues": health_issues,
            "metrics": metrics
        }


class RequestTracer:
    """
    Distributed tracing for RAG requests.

    This class implements request tracing to monitor the performance
    of different components in the RAG pipeline.
    """

    def __init__(self):
        """Initialize request tracer."""
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.trace_lock = threading.Lock()

    def start_trace(self, request_id: str, operation: str) -> str:
        """Start tracing a request."""
        trace_id = f"{request_id}_{operation}_{int(time.time() * 1000)}"

        with self.trace_lock:
            self.traces[trace_id] = {
                "trace_id": trace_id,
                "request_id": request_id,
                "operation": operation,
                "start_time": time.time(),
                "spans": []
            }

        return trace_id

    def add_span(self, trace_id: str, span_name: str, span_data: Dict[str, Any]) -> None:
        """Add a span to an existing trace."""
        with self.trace_lock:
            if trace_id in self.traces:
                span = {
                    "span_name": span_name,
                    "start_time": time.time(),
                    "data": span_data
                }
                self.traces[trace_id]["spans"].append(span)

    def end_trace(self, trace_id: str, status: str = "success", error: Optional[str] = None) -> Dict[str, Any]:
        """End tracing and return trace data."""
        with self.trace_lock:
            if trace_id in self.traces:
                trace = self.traces[trace_id]
                trace["end_time"] = time.time()
                trace["duration"] = trace["end_time"] - trace["start_time"]
                trace["status"] = status
                trace["error"] = error

                # Remove from active traces
                completed_trace = self.traces.pop(trace_id)

                # Log slow traces
                if trace["duration"] > 5.0:  # 5 second threshold
                    logger.warning(f"Slow operation traced: {trace['duration']:.2f}s - {trace['operation']}")

                return completed_trace

        return {}

    def get_trace_summary(self, request_id: str) -> Dict[str, Any]:
        """Get trace summary for a request."""
        # Find all traces for this request
        request_traces = {
            trace_id: trace for trace_id, trace in self.traces.items()
            if trace["request_id"] == request_id
        }

        if not request_traces:
            return {"request_id": request_id, "traces": [], "total_duration": 0}

        total_duration = sum(trace.get("duration", 0) for trace in request_traces.values())

        return {
            "request_id": request_id,
            "trace_count": len(request_traces),
            "total_duration": total_duration,
            "traces": list(request_traces.values())
        }


# Global instances
metrics_collector = MetricsCollector()
request_tracer = RequestTracer()


def trace_operation(operation_name: str) -> Callable:
    """Decorator for tracing operations."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Generate trace ID
            request_id = kwargs.get('request_id', f"req_{int(time.time() * 1000)}")
            trace_id = request_tracer.start_trace(request_id, operation_name)

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # End trace successfully
                request_tracer.end_trace(trace_id, "success")
                return result

            except Exception as e:
                # End trace with error
                request_tracer.end_trace(trace_id, "error", str(e))
                raise

        def sync_wrapper(*args, **kwargs):
            # Generate trace ID
            request_id = kwargs.get('request_id', f"req_{int(time.time() * 1000)}")
            trace_id = request_tracer.start_trace(request_id, operation_name)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # End trace successfully
                request_tracer.end_trace(trace_id, "success")
                return result

            except Exception as e:
                # End trace with error
                request_tracer.end_trace(trace_id, "error", str(e))
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    return metrics_collector.get_health_status()


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return metrics_collector.get_current_metrics()


def main():
    """Example usage of monitoring system."""
    logging.basicConfig(level=logging.INFO)

    try:
        # Get current metrics
        metrics = get_performance_metrics()
        print(f"Current CPU usage: {metrics['system']['cpu_percent']:.1f}%")
        print(f"Current memory usage: {metrics['system']['memory_percent']:.1f}%")

        # Get health status
        health = get_system_health()
        print(f"System health: {health['status']}")
        if health['issues']:
            print(f"Issues: {health['issues']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
