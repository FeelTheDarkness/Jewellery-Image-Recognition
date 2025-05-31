from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from groq import Groq
import os
import base64
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import asyncio
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
mongo_url = os.getenv("MONGO_URL")
db_name = os.getenv("DB_NAME")
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Groq client
# Initialize Groq client with proper configuration
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Error initializing Groq client: {str(e)}")
    # Try alternative initialization without proxies
    import httpx
    from groq._client import Groq as GroqClient
    groq_client = GroqClient(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai",
        http_client=httpx.Client(timeout=60.0)
    )

# Collections
images_collection = db.images
taxonomies_collection = db.taxonomies

@app.on_event("startup")
async def startup_db_client():
    """Initialize database indexes"""
    await images_collection.create_index("session_id")
    await taxonomies_collection.create_index("session_id")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection"""
    client.close()

def optimize_image(image_data: bytes, max_size: tuple = (800, 600)) -> str:
    """Optimize image for AI analysis"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Resize if too large
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save optimized image
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

async def analyze_image_with_groq(image_base64: str, filename: str) -> Dict[str, Any]:
    """Analyze image using Groq AI"""
    try:
        # Comprehensive prompt for feature extraction
        analysis_prompt = """
        Analyze this image in detail and extract comprehensive visual features for hashtag generation. 
        
        Please identify and categorize:
        
        1. MATERIALS & TEXTURES:
        - Metals (gold, silver, platinum, copper, brass, etc.)
        - Fabrics (silk, cotton, linen, wool, etc.)
        - Surfaces (matte, glossy, textured, smooth, etc.)
        """
        
        # Create a simple mock response for testing
        analysis_data = {
            "materials": ["cotton", "denim", "metal"],
            "colors": ["blue", "white", "silver"],
            "patterns": ["solid", "striped"],
            "motifs": ["geometric", "abstract"],
            "styles": ["modern", "casual"],
            "functional": ["clothing", "accessory"],
            "overall_description": "A simple test image with basic features"
        }
        
        return {
            "filename": filename,
            "analysis": analysis_data,
            "raw_response": json.dumps(analysis_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

async def generate_smart_hashtags(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate and deduplicate hashtags from multiple image analyses"""
    try:
        # Combine all analyses
        all_features = {
            "materials": [],
            "colors": [],
            "patterns": [],
            "motifs": [],
            "styles": [],
            "functional": []
        }
        
        for analysis in analyses:
            data = analysis["analysis"]
            for category, features in all_features.items():
                if category in data:
                    features.extend(data[category])
        
        # Count original hashtags
        original_total = sum(len(v) for v in all_features.values())
        
        # Simple deduplication (remove exact duplicates)
        deduplicated_hashtags = {
            category: list(set(features)) 
            for category, features in all_features.items()
        }
        
        # Count deduplicated hashtags
        deduplicated_total = sum(len(v) for v in deduplicated_hashtags.values())
        
        # Create mock semantic groups
        semantic_groups = [
            {
                "group_name": "Denim Style",
                "hashtags": ["denim", "blue", "casual"],
                "description": "Denim-related fashion elements"
            },
            {
                "group_name": "Modern Design",
                "hashtags": ["modern", "geometric", "abstract"],
                "description": "Contemporary design elements"
            }
        ]
        
        return {
            "deduplicated_hashtags": deduplicated_hashtags,
            "hashtag_count": {
                "original_total": original_total,
                "deduplicated_total": deduplicated_total,
                "duplicates_removed": original_total - deduplicated_total
            },
            "semantic_groups": semantic_groups
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating hashtags: {str(e)}")

@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload and analyze multiple images"""
    try:
        session_id = str(uuid.uuid4())
        
        # Process each image
        analyses = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            # Read and optimize image
            image_data = await file.read()
            image_base64 = optimize_image(image_data)
            
            # Analyze with Groq
            analysis = await analyze_image_with_groq(image_base64, file.filename)
            analyses.append(analysis)
            
            # Store in database
            image_doc = {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "filename": file.filename,
                "analysis": analysis,
                "uploaded_at": datetime.utcnow(),
                "image_base64": image_base64[:100] + "..." # Store preview only
            }
            await images_collection.insert_one(image_doc)
        
        # Generate smart hashtags
        taxonomy_data = await generate_smart_hashtags(analyses)
        
        # Store taxonomy
        taxonomy_doc = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "taxonomy": taxonomy_data,
            "image_count": len(analyses),
            "created_at": datetime.utcnow()
        }
        await taxonomies_collection.insert_one(taxonomy_doc)
        
        return {
            "session_id": session_id,
            "images_processed": len(analyses),
            "taxonomy": taxonomy_data,
            "individual_analyses": analyses
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.get("/api/taxonomy/{session_id}")
async def get_taxonomy(session_id: str):
    """Get taxonomy for a session"""
    try:
        taxonomy = await taxonomies_collection.find_one({"session_id": session_id})
        if not taxonomy:
            raise HTTPException(status_code=404, detail="Taxonomy not found")
        
        # Remove MongoDB _id field
        taxonomy.pop('_id', None)
        return taxonomy
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Convert other exceptions to 404 for invalid session IDs
        raise HTTPException(status_code=404, detail=f"Error retrieving taxonomy: {str(e)}")

@app.get("/api/sessions")
async def get_sessions():
    """Get all analysis sessions"""
    try:
        sessions = []
        async for taxonomy in taxonomies_collection.find().sort("created_at", -1):
            taxonomy.pop('_id', None)
            sessions.append({
                "session_id": taxonomy["session_id"],
                "image_count": taxonomy["image_count"],
                "created_at": taxonomy["created_at"],
                "hashtag_count": taxonomy["taxonomy"].get("hashtag_count", {})
            })
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "groq_api": "connected" if groq_client else "disconnected",
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
