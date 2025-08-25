# Geothermal QA Engine

A comprehensive AI-powered question answering system for geothermal energy literature, built with LangGraph orchestration and RAG (Retrieval-Augmented Generation).

## 🎯 **Features**

### **Core Capabilities**
- **Question Matrix Routing**: Uses Excel/CSV matrix to intelligently route questions and select appropriate retrieval strategies
- **Field Detection**: Automatically detects geothermal fields (e.g., Semurup, Kamojang) from questions with typo tolerance
- **Dual RAG System**: Searches both text chunks and image figures from your knowledge base
- **LangGraph Orchestration**: Structured workflow with field detection → intent routing → evidence retrieval → LLM composition
- **Typo-Tolerant Classification**: Uses RapidFuzz for 1-2 character typo tolerance in field names and concepts

### **Smart Retrieval**
- **Pre-filtering**: Uses Question Matrix to build targeted filters before similarity search
- **Progressive Relaxation**: Automatically relaxes filters if no results found (topic → method → aspect → discipline)
- **Evidence Prioritization**: Numeric twin data (future) > literature text > figure captions
- **Efficient Search**: Caption-only search for images, then CLIP similarity if needed

### **Quality Responses** 
- **Varied Language**: Temperature=0.5, frequency_penalty=0.2 to avoid repetitive/template responses
- **Proper Citations**: [doc_id:page] for text, [fig:doc_id:page] for figures
- **Confidence Assessment**: HIGH/MEDIUM/LOW based on evidence quality
- **Graceful Fallbacks**: Honest responses when evidence is insufficient

## 📁 **Architecture**

```
src/
├── config.py              # Configuration and paths
├── api.py                 # FastAPI REST endpoints
├── app_graph.py           # LangGraph workflow orchestration
├── tools/
│   ├── qm.py             # Question Matrix loader & router
│   ├── field_detect.py   # Typo-tolerant field detection
│   ├── rag_text.py       # Text RAG operations
│   ├── rag_image.py      # Image RAG operations
│   └── ...
├── prompts/
│   └── system_prompt.py  # LLM prompting logic
└── ...

tests/
└── smoke_qm.py           # Comprehensive acceptance tests

ingest_guide/
└── Question Matrix.csv   # Question routing configuration
```

## 🚀 **Quick Start**

### **1. Prerequisites**
- Python 3.8+
- OpenAI API key
- Existing Chroma knowledge base (from ingestion pipeline)

### **2. Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# Windows PowerShell:
$env:OPENAI_API_KEY="your-api-key-here"
```

### **3. Test Components**
```bash
# Test all components (no API key needed)
python test_qa_demo.py
```

### **4. Start API Server**
```bash
uvicorn src.api:app --reload
```

### **5. Test Questions**

#### **With Field Detection**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Where is the caprock and how thick is it?","field":"Semurup"}'
```

#### **General Questions** 
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain typical caprock lithologies in Indonesian volcanic settings."}'
```

#### **PowerShell Examples**
```powershell
# With field
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question":"What geothermometers say about reservoir temperature?","field":"Semurup"}'

# General question
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question":"How do geothermometers work in volcanic geothermal systems?"}'
```

## 📋 **API Endpoints**

### **POST /ask**
Ask a question to the QA engine.

**Request:**
```json
{
  "question": "Where is the caprock in Semurup?",
  "field": "Semurup"  // optional
}
```

**Response:**
```json
{
  "answer": "Based on the geological evidence, the caprock in Semurup...",
  "citations": ["doc_id:page", "fig:doc_id:page"],
  "figures": [{"caption": "...", "relative_path": "..."}],
  "field": "Semurup",
  "intent": "caprock_inquiry", 
  "confidence": "HIGH",
  "text_chunks_found": 5,
  "figures_found": 2
}
```

### **GET /fields**
Get available fields from knowledge base.

### **GET /health** 
Health check with component status.

### **GET /stats**
Knowledge base statistics.

## ⚙️ **Configuration**

### **Question Matrix Format** (`ingest_guide/Question Matrix.csv`)
```csv
id,user_question,intent_tag,primary_aspect,secondary_aspects,discipline_hints,method_or_topic_hints,requires_twin,retrieval_filter_hint,expected_outputs,eval_keywords,priority
1,"Where is the caprock?",caprock_inquiry,Caprock,"[""Permeability""]","[""Geology""]","[""MT""]",false,"{""aspect"": ""Caprock""}","[""depth"", ""thickness""]","[""caprock"", ""seal""]",1
```

### **Environment Variables**
```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional path overrides
QM_PATH=/path/to/question-matrix.csv
TEXT_DB_PATH=/path/to/text_emb
IMAGE_DB_PATH=/path/to/image_emb
IMAGES_DIR_PATH=/path/to/images
```

### **Model Configuration** (`src/config.py`)
```python
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.5
LLM_FREQUENCY_PENALTY = 0.2
TEXT_RETRIEVAL_TOP_K = 6
IMAGE_RETRIEVAL_TOP_K = 3
```

## 🧪 **Testing**

### **Component Tests**
```bash
# Test individual components
python test_qa_demo.py
```

### **Smoke Tests**  
```bash
# Test via API (requires server running)
python tests/smoke_qm.py --api-key your-key-here

# Test direct workflow calls only
python tests/smoke_qm.py --api-key your-key-here --no-api

# Save results to file
python tests/smoke_qm.py --api-key your-key-here --output results.json
```

### **Expected Test Results**
- ✅ Question Matrix loads 8+ patterns
- ✅ Field detection identifies "Semurup" from questions
- ✅ Intent inference with >70% confidence  
- ✅ Text/image retrieval finds relevant chunks
- ✅ Integration workflow generates proper filters
- ✅ All API endpoints respond correctly

## 🎯 **Sample Questions**

### **Field-Specific**
- "Where is the caprock and how thick is it?" (Semurup)
- "What is the reservoir temperature based on geothermometers?" (Semurup)
- "How does MT data show the subsurface structure?" (Semurup) 
- "What does the hydrology tell us about the geothermal system?" (Semurup)
- "Where should we target wells for development?" (Semurup)

### **General**
- "Explain typical caprock lithologies in Indonesian volcanic settings"
- "How do geothermometers work in volcanic geothermal systems?"
- "What are the main geophysical methods for geothermal exploration?"

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"Collection expecting embedding with dimension X, got Y"**
- **Cause**: Existing Chroma collections were created with different embedding models
- **Solution**: Re-run ingestion pipeline or delete existing collections

#### **"No evidence found"**
- **Cause**: Filters too restrictive or insufficient knowledge base
- **Solution**: Check Question Matrix filters, verify data ingestion

#### **"OpenAI API key not configured"**
- **Solution**: Set `OPENAI_API_KEY` environment variable

#### **Filter syntax errors**
- **Cause**: Chroma filter format issues  
- **Solution**: Updated in latest version, restart API server

### **Debug Mode**
```bash
# Enable debug logging
python test_qa_demo.py --log-level DEBUG

# Check API health
curl http://127.0.0.1:8000/health
```

## 🔮 **Future Extensions**

### **Planned Features**
- **Digital Twin Integration**: Numeric data from simulation models
- **Vision LLM**: Advanced figure analysis with vision models
- **Multi-language**: Indonesian language support
- **Caching**: Redis caching for faster responses
- **Streaming**: Real-time response streaming

### **Scaffold Hooks**
- `numeric_ctx` placeholder in workflow for twin data
- `vision_budget` for selective figure analysis
- Confidence assessment framework for answer quality

## 📊 **Performance**

### **Typical Response Times**
- Field detection: ~50ms
- Intent inference: ~200ms  
- Text retrieval: ~300ms
- Image retrieval: ~200ms
- LLM composition: ~2-5s
- **Total**: ~3-6 seconds

### **Optimization Features**
- Cached Question Matrix and field lists
- Pre-filtering before similarity search
- Progressive filter relaxation
- Small top-k limits (text=6, images=3)
- Efficient caption-first image search

---

## 🎉 **Ready to Use!**

The QA engine is now fully functional and ready for production use. Start with the component tests, then try the API endpoints with your OpenAI key. The system will intelligently route questions, retrieve relevant evidence, and generate high-quality, well-cited responses about geothermal energy topics.

**Happy questioning! 🔥⚡**
