# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint']
)

GENERATION_QUALITY = Gauge(
    'generation_quality_score',
    'Quality score of generated content (0-1)'
)

THROUGHPUT = Gauge(
    'api_throughput_requests_per_second',
    'Current throughput in requests per second'
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

# Request/Response models
class ProductRequest(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Product description")
    style: str = Field(default="professional", description="Ad style: professional, casual, creative")

class AdCreativeResponse(BaseModel):
    product_id: str
    ad_text: str
    generation_time: float
    quality_score: float
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Ad Creative Generator",
    description="AI-powered advertisement creative generation API",
    version="1.0.0"
)

# Global model variables
model = None
tokenizer = None
model_version = "1.0.0"

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer
    start_time = time.time()
    
    try:
        model_path = "models/checkpoints/final_model"  # Can be overridden by env variable
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def calculate_quality_score(generated_text: str) -> float:
    """Calculate quality score for generated ad"""
    # Simple quality metrics
    score = 0.0
    
    # Length check (should be between 50-200 chars)
    length = len(generated_text)
    if 50 <= length <= 200:
        score += 0.3
    elif length > 0:
        score += 0.15
    
    # Check for emoji (engaging)
    if any(char in generated_text for char in '🚀✨💎🔥⚡🎉'):
        score += 0.2
    
    # Check for call-to-action words
    cta_words = ['shop', 'buy', 'order', 'get', 'discover', 'limited', 'now', 'today']
    if any(word in generated_text.lower() for word in cta_words):
        score += 0.3
    
    # Check capitalization (not all caps)
    if not generated_text.isupper():
        score += 0.2
    
    return min(score, 1.0)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=AdCreativeResponse)
async def generate_ad_creative(request: ProductRequest):
    """Generate ad creative for a product"""
    start_time = time.time()
    
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Construct input
        input_text = f"Generate ad for: {request.product_name}. Category: {request.category}. Description: {request.description}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        quality_score = calculate_quality_score(generated_text)
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/generate").observe(generation_time)
        GENERATION_QUALITY.set(quality_score)
        
        logger.info(f"Generated ad for {request.product_id} in {generation_time:.3f}s")
        
        return AdCreativeResponse(
            product_id=request.product_id,
            ad_text=generated_text,
            generation_time=generation_time,
            quality_score=quality_score,
            model_version=model_version
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="500").inc()
        logger.error(f"Error generating ad: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-generate")
async def batch_generate(products: list[ProductRequest], background_tasks: BackgroundTasks):
    """Generate ads for multiple products"""
    results = []
    
    for product in products:
        try:
            result = await generate_ad_creative(product)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to generate ad for {product.product_id}: {e}")
            results.append({
                "product_id": product.product_id,
                "error": str(e)
            })
    
    return {"results": results, "total": len(products), "successful": len([r for r in results if "error" not in r])}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "E-Commerce Ad Creative Generator",
        "version": model_version,
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "batch": "/batch-generate",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)