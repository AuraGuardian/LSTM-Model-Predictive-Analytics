from fastapi import FastAPI
from prometheus_client import make_asgi_app, generate_latest, CONTENT_TYPE_LATEST
from monitoring import monitor
import uvicorn

app = FastAPI()

# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "ML Model Metrics Server is running. Access /metrics for Prometheus metrics."
    }

def start_metrics_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the metrics server"""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
    server = uvicorn.Server(config)
    return server

if __name__ == "__main__":
    # Start the metrics server
    server = start_metrics_server()
    server.run()
