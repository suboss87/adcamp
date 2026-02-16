"""
Monitoring and Observability
Prometheus metrics, health checks, and observability utilities.
"""
from typing import Dict, Any
from datetime import datetime

# In-memory metrics store (use Redis/Prometheus in production)
_metrics = {
    "videos_generated_total": 0,
    "videos_failed_total": 0,
    "api_requests_total": 0,
    "script_generation_duration_seconds": [],
    "video_generation_duration_seconds": [],
    "total_cost_usd": 0.0,
    "hero_videos": 0,
    "catalog_videos": 0,
}


def increment_counter(metric: str, value: int = 1):
    """Increment a counter metric."""
    if metric in _metrics:
        _metrics[metric] += value


def record_duration(metric: str, duration_seconds: float):
    """Record a duration metric."""
    if metric in _metrics and isinstance(_metrics[metric], list):
        _metrics[metric].append(duration_seconds)
        # Keep only last 1000 samples
        if len(_metrics[metric]) > 1000:
            _metrics[metric] = _metrics[metric][-1000:]


def add_cost(cost_usd: float):
    """Add to total cost."""
    _metrics["total_cost_usd"] += cost_usd


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot."""
    # Calculate averages for duration metrics
    script_avg = (
        sum(_metrics["script_generation_duration_seconds"])
        / len(_metrics["script_generation_duration_seconds"])
        if _metrics["script_generation_duration_seconds"]
        else 0
    )
    video_avg = (
        sum(_metrics["video_generation_duration_seconds"])
        / len(_metrics["video_generation_duration_seconds"])
        if _metrics["video_generation_duration_seconds"]
        else 0
    )

    return {
        "videos_generated_total": _metrics["videos_generated_total"],
        "videos_failed_total": _metrics["videos_failed_total"],
        "api_requests_total": _metrics["api_requests_total"],
        "script_generation_avg_seconds": round(script_avg, 2),
        "video_generation_avg_seconds": round(video_avg, 2),
        "total_cost_usd": round(_metrics["total_cost_usd"], 4),
        "avg_cost_per_video": (
            round(_metrics["total_cost_usd"] / _metrics["videos_generated_total"], 4)
            if _metrics["videos_generated_total"] > 0
            else 0
        ),
        "hero_videos": _metrics["hero_videos"],
        "catalog_videos": _metrics["catalog_videos"],
    }


def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": get_metrics(),
        "uptime_seconds": "N/A",  # Would need startup time tracking
    }


def prometheus_format() -> str:
    """Format metrics in Prometheus text format."""
    metrics = get_metrics()
    lines = [
        "# HELP videos_generated_total Total number of videos generated",
        "# TYPE videos_generated_total counter",
        f"videos_generated_total {metrics['videos_generated_total']}",
        "",
        "# HELP videos_failed_total Total number of failed video generations",
        "# TYPE videos_failed_total counter",
        f"videos_failed_total {metrics['videos_failed_total']}",
        "",
        "# HELP api_requests_total Total number of API requests",
        "# TYPE api_requests_total counter",
        f"api_requests_total {metrics['api_requests_total']}",
        "",
        "# HELP script_generation_avg_seconds Average script generation duration",
        "# TYPE script_generation_avg_seconds gauge",
        f"script_generation_avg_seconds {metrics['script_generation_avg_seconds']}",
        "",
        "# HELP video_generation_avg_seconds Average video generation duration",
        "# TYPE video_generation_avg_seconds gauge",
        f"video_generation_avg_seconds {metrics['video_generation_avg_seconds']}",
        "",
        "# HELP total_cost_usd Total cost in USD",
        "# TYPE total_cost_usd gauge",
        f"total_cost_usd {metrics['total_cost_usd']}",
        "",
        "# HELP avg_cost_per_video Average cost per video in USD",
        "# TYPE avg_cost_per_video gauge",
        f"avg_cost_per_video {metrics['avg_cost_per_video']}",
        "",
        "# HELP hero_videos_total Total hero videos generated",
        "# TYPE hero_videos_total counter",
        f"hero_videos_total {metrics['hero_videos']}",
        "",
        "# HELP catalog_videos_total Total catalog videos generated",
        "# TYPE catalog_videos_total counter",
        f"catalog_videos_total {metrics['catalog_videos']}",
        "",
    ]
    return "\n".join(lines)
