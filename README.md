# 🇮🇳 Government Scheme Eligibility & Application Readiness Copilot

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://github.com/langchain-ai/langgraph)

> **Empowering rural India through AI-powered government scheme discovery and application assistance**

An AI-powered, voice-first, multilingual assistant that bridges the accessibility gap between rural Indian citizens and government welfare schemes. Built with modern AI architecture including LLMs, RAG, agent-based orchestration, and document intelligence.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Agent System](#-agent-system)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

India operates hundreds of government welfare schemes, yet scheme utilization remains suboptimal due to awareness gaps, accessibility barriers, and documentation challenges. This system addresses these issues by providing:

- **Conversational Scheme Discovery**: Natural language queries to find relevant schemes
- **Explainable Eligibility Assessment**: Transparent reasoning for eligibility decisions
- **Documentation Validation**: AI-powered document verification using OCR
- **Application Readiness**: Generate structured, prefilled application data
- **Multilingual Support**: Hindi, English, and major Indian regional languages
- **Voice-First Interface**: Accessible to low-literacy users

### 🎯 Target Impact

- Enable millions of rural citizens to discover relevant schemes
- Reduce dependency on intermediaries
- Increase scheme enrollment rates
- Provide transparent, explainable eligibility reasoning

---

## ✨ Key Features

### 🗣️ Multimodal Interaction
- **Text Chat**: Natural language conversation in multiple languages
- **Voice Input**: Speech-to-text with regional accent support
- **Document Upload**: Image and PDF processing with OCR

### 🤖 AI-Powered Intelligence
- **Context Extraction**: Automatically extract user attributes from conversation
- **Semantic Search**: RAG-based scheme retrieval with hybrid search
- **Eligibility Validation**: Hybrid rule-based + LLM validation
- **Document Intelligence**: OCR, classification, and field extraction

### 🔍 Explainability
- Transparent reasoning for all eligibility decisions
- Confidence scores for AI-generated outputs
- Criterion-level explanations
- "What-if" scenario analysis

### 🌐 Multilingual Support
- Hindi, English, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi
- Automatic language detection
- Regional scheme prioritization

### 📄 Application Readiness
- Structured JSON application data
- Prefilled forms with user context
- Document validation checklist
- Step-by-step submission guide

---

## 🏗️ Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Web App    │  │  Mobile Web  │  │Voice Interface│         │
│  │   (React)    │  │  (Responsive)│  │  (STT/TTS)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                              │
│              (Authentication, Rate Limiting, Routing)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Agent Orchestration Engine                   │  │
│  │                    (LangGraph)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────┬───────────┬──────────┬──────────┬──────────┐   │
│  │  Context  │  Scheme   │Eligibility│Document  │App Ready │   │
│  │  Builder  │ Researcher│ Validator │  Agent   │  Agent   │   │
│  │   Agent   │   Agent   │   Agent   │          │          │   │
│  └───────────┴───────────┴──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVICE LAYER                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   LLM    │ │   RAG    │ │   OCR    │ │ Validation│          │
│  │ Service  │ │ Service  │ │ Service  │ │  Service  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │PostgreSQL│ │  Vector  │ │  Object  │ │  Cache   │          │
│  │   (User  │ │   DB     │ │ Storage  │ │  (Redis) │          │
│  │  Profile)│ │ (Schemes)│ │  (Docs)  │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Agent-Based Architecture

The system uses **LangGraph** for agent orchestration with specialized agents:

1. **Context Builder Agent**: Extracts structured user attributes from conversation
2. **Scheme Researcher Agent**: Retrieves relevant schemes using RAG pipeline
3. **Eligibility Validator Agent**: Validates eligibility using hybrid approach
4. **Documentation Agent**: Processes and validates uploaded documents
5. **Application Readiness Agent**: Generates application-ready outputs

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Agent Orchestration**: LangGraph
- **LLM**: OpenAI GPT-4 / Google Gemini
- **Vector Database**: Pinecone / Weaviate / Chroma
- **Database**: PostgreSQL 15+
- **Cache**: Redis 7+
- **OCR**: Tesseract / Google Vision API
- **Task Queue**: Celery

### Frontend
- **Framework**: React 18+
- **UI Library**: Material-UI / Tailwind CSS
- **State Management**: Redux Toolkit
- **API Client**: Axios
- **Voice**: Web Speech API

### Infrastructure
- **Cloud**: AWS / GCP
- **Containers**: Docker, ECS/Cloud Run
- **IaC**: Terraform
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana, Sentry

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/scheme-copilot.git
cd scheme-copilot
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/scheme_copilot
REDIS_URL=redis://localhost:6379/0

# LLM
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4-turbo

# Vector Database
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env

# Storage
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=your_bucket_name

# Security
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

4. **Initialize database**
```bash
# Run migrations
alembic upgrade head

# Seed initial data
python scripts/seed_schemes.py
```

5. **Generate embeddings for schemes**
```bash
python scripts/generate_embeddings.py
```

6. **Start the backend server**
```bash
uvicorn src.api.main:app --reload --port 8000
```

7. **Start the frontend (in a new terminal)**
```bash
cd frontend
npm install
npm start
```

8. **Access the application**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- API Redoc: http://localhost:8000/redoc

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📁 Project Structure

```
scheme-copilot/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API entry point
│   │   ├── routes/            # API endpoints
│   │   ├── middleware/        # Custom middleware
│   │   └── dependencies.py    # Dependency injection
│   ├── agents/                # Agent implementations
│   │   ├── context_builder.py
│   │   ├── scheme_researcher.py
│   │   ├── eligibility_validator.py
│   │   ├── documentation.py
│   │   └── app_readiness.py
│   ├── services/              # Business logic services
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   ├── ocr_service.py
│   │   └── validation_service.py
│   ├── models/                # Data models
│   │   ├── user.py
│   │   ├── scheme.py
│   │   └── document.py
│   ├── db/                    # Database layer
│   │   ├── repositories/
│   │   └── migrations/
│   └── utils/                 # Utility functions
├── frontend/                  # React application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── store/
│   └── public/
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/                   # Utility scripts
│   ├── seed_schemes.py
│   └── generate_embeddings.py
├── config/                    # Configuration files
├── docs/                      # Documentation
│   ├── requirements.md
│   ├── design.md
│   └── api.md
├── docker/                    # Docker configurations
├── terraform/                 # Infrastructure as Code
├── .github/                   # GitHub workflows
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📚 API Documentation

### Authentication

All API endpoints (except `/auth/*`) require JWT authentication.

**Register/Login**
```bash
POST /api/v1/auth/register
{
  "phone_number": "9876543210",
  "language_preference": "hi"
}

POST /api/v1/auth/verify-otp
{
  "phone_number": "9876543210",
  "otp": "123456"
}
```

### Chat Interface

**Send Message**
```bash
POST /api/v1/chat/message
Authorization: Bearer <token>
{
  "session_id": "uuid",
  "message": "I am a farmer from Maharashtra",
  "language": "en"
}
```

**Voice Input**
```bash
POST /api/v1/chat/voice
Authorization: Bearer <token>
Content-Type: multipart/form-data

session_id: uuid
audio_file: <binary>
language: hi
```

### Scheme Discovery

**Search Schemes**
```bash
GET /api/v1/schemes/search?query=farmer&state=Maharashtra&limit=10
Authorization: Bearer <token>
```

**Check Eligibility**
```bash
POST /api/v1/schemes/{scheme_id}/check-eligibility
Authorization: Bearer <token>
{
  "session_id": "uuid"
}
```

### Document Management

**Upload Documents**
```bash
POST /api/v1/documents/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

session_id: uuid
files: [<binary>, <binary>]
document_types: ["aadhaar", "income_certificate"]
```

**Validate Documents**
```bash
POST /api/v1/documents/validate
Authorization: Bearer <token>
{
  "session_id": "uuid",
  "document_ids": ["uuid1", "uuid2"],
  "scheme_id": "uuid"
}
```

### Application Generation

**Generate Application**
```bash
POST /api/v1/applications/generate
Authorization: Bearer <token>
{
  "session_id": "uuid",
  "scheme_id": "uuid"
}
```

For complete API documentation, visit `/docs` after starting the server.

---

## 🤖 Agent System

### Agent Workflow

```
User Input → Context Builder → Scheme Researcher → Eligibility Validator
                                                            ↓
                                                    Documentation Agent
                                                            ↓
                                                    Application Readiness
```

### Context Builder Agent

Extracts structured user attributes from conversational input:
- Demographics (age, gender, location)
- Economic status (income, occupation)
- Social category (caste, minority status)
- Special categories (disability, BPL status)

**Confidence Scoring**:
- High (>0.8): Explicitly stated
- Medium (0.5-0.8): Inferred from context
- Low (<0.5): Assumed or missing

### Scheme Researcher Agent

Uses RAG pipeline for scheme retrieval:
1. Build search query from user context
2. Perform hybrid search (keyword + semantic)
3. Retrieve top-k schemes from vector database
4. Rerank based on user context match
5. Generate retrieval reasoning

### Eligibility Validator Agent

Hybrid validation approach:
- **Rule-based**: For explicit criteria (age, income thresholds)
- **LLM-based**: For complex/ambiguous criteria
- **Explainable**: Provides reasoning for each criterion

### Documentation Agent

Document processing pipeline:
1. OCR text extraction
2. Document type classification
3. Structured field extraction
4. Validation against user context
5. Completeness check

### Application Readiness Agent

Generates application-ready outputs:
- Structured JSON with all required fields
- Prefilled form data
- Document checklist
- Step-by-step submission guide

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build images
docker build -t scheme-copilot-api -f docker/Dockerfile.api .
docker build -t scheme-copilot-frontend -f docker/Dockerfile.frontend .

# Run containers
docker-compose up -d
```

### AWS ECS Deployment

```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
cd terraform
terraform init
terraform plan
terraform apply

# Deploy application
./scripts/deploy.sh production
```

### Environment-Specific Configuration

- **Development**: `config/development.yaml`
- **Staging**: `config/staging.yaml`
- **Production**: `config/production.yaml`

### Monitoring

- **Metrics**: Prometheus + Grafana
- **Logging**: CloudWatch / Stackdriver
- **Tracing**: OpenTelemetry
- **Error Tracking**: Sentry

---

## 🧪 Testing

### Run Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# E2E tests
pytest tests/e2e -v

# Coverage report
pytest --cov=src --cov-report=html
```

### Test Coverage Goals

- Unit tests: >80%
- Integration tests: >70%
- E2E tests: Critical user flows

---

## 📊 Performance Metrics

### Target Metrics (MVP)

- **Response Time**: <3 seconds for chat interactions
- **Voice Processing**: <2 seconds for speech-to-text
- **Document OCR**: <10 seconds per document
- **Scheme Retrieval**: <2 seconds
- **Concurrent Users**: 1000+
- **Uptime**: 99%

### AI Model Performance

- **Context Extraction**: >85% accuracy
- **Scheme Retrieval**: >80% relevance (top-5)
- **Eligibility Validation**: >90% accuracy
- **Document Classification**: >85% accuracy
- **OCR Accuracy**: >90%

---

## 🗺️ Roadmap

### Phase 1: MVP (Current)
- ✅ Core agent architecture
- ✅ Text-based chat interface
- ✅ Voice input support
- ✅ Document upload and OCR
- ✅ Eligibility validation
- ✅ Application readiness generation
- ✅ Multilingual support (4 languages)

### Phase 2: Enhancement
- [ ] Operator dashboard (human-in-the-loop)
- [ ] Application status tracking
- [ ] SMS and WhatsApp integration
- [ ] Expanded scheme database (500+ schemes)
- [ ] Support for 20+ Indian languages
- [ ] Mobile app (Android)
- [ ] Advanced document validation

### Phase 3: Scale
- [ ] Government portal integration (if APIs available)
- [ ] Real-time application status tracking
- [ ] Video KYC integration
- [ ] Biometric authentication
- [ ] Blockchain-based document verification
- [ ] Predictive analytics

### Long-Term Vision
- [ ] National-scale deployment (all states)
- [ ] Integration with India Stack (Aadhaar, DigiLocker)
- [ ] Partnership with government for official adoption
- [ ] Expansion to other government services
- [ ] Open-source platform for other countries

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Python: Follow PEP 8, use `black` for formatting
- JavaScript: Follow Airbnb style guide, use `prettier`
- Commit messages: Follow Conventional Commits

### Testing Requirements

- All new features must include unit tests
- Integration tests for API endpoints
- E2E tests for critical user flows

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**TheBugBangTheory** - AI4Bharat Hackathon 2026

- [Team Member 1] - Role
- [Team Member 2] - Role
- [Team Member 3] - Role

---

## 🙏 Acknowledgments

- **AI4Bharat** for organizing the hackathon
- **OpenAI** for GPT-4 API
- **LangChain** for LangGraph framework
- **Government of India** for scheme data
- All contributors and supporters

---

## 📞 Contact

- **Email**: contact@scheme-copilot.in
- **Website**: https://scheme-copilot.in
- **Twitter**: [@SchemeCopilot](https://twitter.com/schemecopilot)
- **Discord**: [Join our community](https://discord.gg/schemecopilot)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/scheme-copilot&type=Date)](https://star-history.com/#yourusername/scheme-copilot&Date)

---

<div align="center">

**Made with ❤️ for Rural India**

[Report Bug](https://github.com/yourusername/scheme-copilot/issues) · [Request Feature](https://github.com/yourusername/scheme-copilot/issues) · [Documentation](https://docs.scheme-copilot.in)

</div>
