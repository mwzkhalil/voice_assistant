"""
Prometheus metrics endpoint.

GET /metrics - Returns Prometheus text format metrics.
"""

from __future__ import annotations

import torch
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from core.metrics import MetricsCollector

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
)
async def metrics() -> Response:
    """
    Expose all Prometheus metrics in text format.

    This endpoint is scraped by Prometheus at configured intervals.
    """
    # Update GPU metrics if available
    if torch.cuda.is_available():
        try:
            for device_idx in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(device_idx)
                free, total = mem_info
                used = total - free
                MetricsCollector.set_gpu_memory(used, total)
        except Exception:
            pass

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
