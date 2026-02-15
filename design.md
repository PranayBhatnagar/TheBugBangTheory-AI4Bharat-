# Design Document: AI-Powered Government Scheme Eligibility & Application Readiness Copilot

## 1. System Architecture Overview

This system implements a multi-agent AI architecture for government scheme discovery, eligibility validation, and application readiness generation. The architecture is designed for modularity, scalability, and explainability.

### 1.1 Architecture Principles

- **Agent-Based Design**: Specialized agents handle distinct responsibilities
- **RAG-First Approach**: Knowledge retrieval augments LLM reasoning
- **Hybrid Validation**: Combine rule-based and AI-based validation
- **Explainability by Design**: All decisions include reasoning traces
- **Human-in-the-Loop Ready**: Support for assisted workflows
- **Privacy-First**: Encryption, consent, and data minimization

### 1.2 Key Architectural Patterns

- **Microservices Architecture**: Independent, scalable services
- **Event-Driven Communication**: Asynchronous agent coordination
- **CQRS Pattern**: Separate read and write operations
- **Repository Pattern**: Abstract data access layer
- **Strategy Pattern**: Pluggable validation and reasoning strategies

## 2. High-Level Architecture Description

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
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  OpenAI  │ │  Google  │ │  Twilio  │ │   SMS    │          │
│  │   API    │ │  Gemini  │ │  (Voice) │ │ Gateway  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Layer Responsibilities

**Presentation Layer**:
- User interface rendering
- Input capture (text, voice, documents)
- Output presentation (text, speech, visualizations)

**API Gateway**:
- Request routing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation

**Application Layer**:
- Agent orchestration and coordination
- Business logic execution
- State management
- Decision-making workflows

**Service Layer**:
- Specialized AI services (LLM, RAG, OCR)
- Validation services
- Integration services

**Data Layer**:
- Persistent storage
- Caching
- Document storage

**External Services**:
- Third-party AI APIs
- Communication services


## 3. Component-Level Design

### 3.1 Frontend Components

#### 3.1.1 Chat Interface Component
```typescript
interface ChatInterfaceProps {
  userId: string;
  sessionId: string;
  language: string;
}

// Features:
// - Message history display
// - Text input with multilingual support
// - Voice input button
// - Document upload widget
// - Scheme card renderer
// - Typing indicators
```

#### 3.1.2 Voice Interface Component
```typescript
interface VoiceInterfaceProps {
  onTranscript: (text: string) => void;
  language: string;
  isListening: boolean;
}

// Features:
// - Speech-to-text integration
// - Text-to-speech playback
// - Audio visualization
// - Language selection
```

#### 3.1.3 Document Upload Component
```typescript
interface DocumentUploadProps {
  onUpload: (files: File[]) => void;
  maxFiles: number;
  acceptedFormats: string[];
}

// Features:
// - Drag-and-drop upload
// - Image preview
// - Upload progress
// - Validation feedback
```

### 3.2 Backend Components

#### 3.2.1 API Gateway Service
```python
# FastAPI application
# Responsibilities:
# - JWT authentication
# - Request validation
# - Rate limiting (per user/IP)
# - CORS handling
# - Request logging
# - Error handling

class APIGateway:
    def __init__(self):
        self.auth_service = AuthService()
        self.rate_limiter = RateLimiter()
        
    async def authenticate(self, token: str) -> User:
        pass
        
    async def route_request(self, request: Request) -> Response:
        pass
```

#### 3.2.2 Session Manager
```python
class SessionManager:
    """Manages user conversation sessions"""
    
    def create_session(self, user_id: str) -> Session:
        """Create new conversation session"""
        pass
        
    def get_session(self, session_id: str) -> Session:
        """Retrieve existing session"""
        pass
        
    def update_context(self, session_id: str, context: dict):
        """Update session context"""
        pass
        
    def expire_session(self, session_id: str):
        """Expire inactive session"""
        pass
```

#### 3.2.3 LLM Service
```python
class LLMService:
    """Wrapper for LLM API calls"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.client = self._init_client()
        
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: dict = None
    ) -> LLMResponse:
        """Generate LLM response"""
        pass
        
    async def generate_structured(
        self,
        prompt: str,
        schema: dict
    ) -> dict:
        """Generate structured JSON response"""
        pass
        
    def _handle_error(self, error: Exception):
        """Error handling with fallback"""
        pass
```

#### 3.2.4 RAG Service
```python
class RAGService:
    """Retrieval-Augmented Generation service"""
    
    def __init__(self, vector_db: VectorDB, embedding_model: str):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict = None
    ) -> List[Document]:
        """Retrieve relevant documents"""
        pass
        
    async def hybrid_search(
        self,
        query: str,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Document]:
        """Hybrid keyword + semantic search"""
        pass
        
    async def rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """Rerank retrieved documents"""
        pass
```

#### 3.2.5 OCR Service
```python
class OCRService:
    """Document OCR and parsing service"""
    
    def __init__(self, ocr_engine: str = "tesseract"):
        self.ocr_engine = ocr_engine
        
    async def extract_text(
        self,
        image: bytes,
        language: str = "eng+hin"
    ) -> OCRResult:
        """Extract text from image"""
        pass
        
    async def classify_document(
        self,
        text: str,
        image: bytes
    ) -> DocumentType:
        """Classify document type"""
        pass
        
    async def extract_fields(
        self,
        text: str,
        document_type: DocumentType
    ) -> dict:
        """Extract structured fields"""
        pass
```

#### 3.2.6 Validation Service
```python
class ValidationService:
    """Rule-based validation service"""
    
    def validate_age(self, age: int, criteria: dict) -> ValidationResult:
        """Validate age criteria"""
        pass
        
    def validate_income(self, income: float, criteria: dict) -> ValidationResult:
        """Validate income criteria"""
        pass
        
    def validate_location(self, location: str, criteria: dict) -> ValidationResult:
        """Validate location criteria"""
        pass
        
    def validate_document(self, document: dict, requirements: dict) -> ValidationResult:
        """Validate document completeness"""
        pass
```


## 4. Agent Architecture Design

### 4.1 Agent Framework

The system uses **LangGraph** for agent orchestration, providing:
- State management across agent execution
- Conditional routing between agents
- Error handling and recovery
- Observability and tracing

### 4.2 Agent Definitions

#### 4.2.1 User Context Builder Agent

**Purpose**: Extract structured user attributes from conversational input

**Input**:
```python
{
    "user_message": str,
    "conversation_history": List[Message],
    "current_context": UserContext
}
```

**Output**:
```python
{
    "updated_context": UserContext,
    "confidence_scores": dict,
    "missing_attributes": List[str],
    "clarification_questions": List[str]
}
```

**Logic Flow**:
1. Analyze user message using LLM
2. Extract entities (age, location, income, etc.)
3. Assign confidence scores
4. Merge with existing context
5. Identify missing critical attributes
6. Generate clarification questions if needed

**Prompt Template**:
```python
CONTEXT_EXTRACTION_PROMPT = """
You are a user context extraction agent. Extract structured user attributes from the conversation.

Conversation History:
{conversation_history}

Current User Context:
{current_context}

New User Message:
{user_message}

Extract the following attributes with confidence scores (0-1):
- age: int
- gender: str
- location: str (state, district)
- income: float (annual)
- occupation: str
- caste_category: str (General/OBC/SC/ST)
- disability: bool
- bpl_status: bool
- land_ownership: float (acres)
- household_size: int

Output JSON format:
{{
    "extracted_attributes": {{}},
    "confidence_scores": {{}},
    "missing_critical": []
}}
"""
```

**Implementation**:
```python
class ContextBuilderAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
    async def execute(self, state: AgentState) -> AgentState:
        prompt = self._build_prompt(state)
        response = await self.llm_service.generate_structured(
            prompt=prompt,
            schema=CONTEXT_SCHEMA
        )
        
        updated_context = self._merge_context(
            state.user_context,
            response["extracted_attributes"],
            response["confidence_scores"]
        )
        
        state.user_context = updated_context
        state.missing_attributes = response["missing_critical"]
        
        return state
```

#### 4.2.2 Scheme Researcher Agent

**Purpose**: Retrieve relevant schemes using RAG pipeline

**Input**:
```python
{
    "user_context": UserContext,
    "user_query": str,
    "filters": dict  # state, category, etc.
}
```

**Output**:
```python
{
    "schemes": List[Scheme],
    "relevance_scores": List[float],
    "retrieval_reasoning": str
}
```

**Logic Flow**:
1. Build search query from user context
2. Perform hybrid search (keyword + semantic)
3. Retrieve top-k schemes
4. Rerank based on user context match
5. Generate retrieval reasoning

**Implementation**:
```python
class SchemeResearcherAgent:
    def __init__(self, rag_service: RAGService, llm_service: LLMService):
        self.rag_service = rag_service
        self.llm_service = llm_service
        
    async def execute(self, state: AgentState) -> AgentState:
        # Build search query
        search_query = self._build_search_query(state.user_context)
        
        # Retrieve schemes
        documents = await self.rag_service.hybrid_search(
            query=search_query,
            top_k=10
        )
        
        # Rerank based on context
        reranked = await self._rerank_by_context(
            documents,
            state.user_context
        )
        
        # Generate reasoning
        reasoning = await self._generate_reasoning(
            state.user_context,
            reranked[:5]
        )
        
        state.retrieved_schemes = reranked[:5]
        state.retrieval_reasoning = reasoning
        
        return state
        
    def _build_search_query(self, context: UserContext) -> str:
        """Build search query from user context"""
        query_parts = []
        
        if context.occupation:
            query_parts.append(f"schemes for {context.occupation}")
        if context.location:
            query_parts.append(f"in {context.location}")
        if context.income and context.income < 100000:
            query_parts.append("low income")
        if context.caste_category in ["SC", "ST", "OBC"]:
            query_parts.append(f"{context.caste_category} category")
            
        return " ".join(query_parts)
```

#### 4.2.3 Eligibility Validator Agent

**Purpose**: Validate user eligibility for schemes using hybrid approach

**Input**:
```python
{
    "user_context": UserContext,
    "scheme": Scheme
}
```

**Output**:
```python
{
    "is_eligible": bool,
    "confidence": float,
    "satisfied_criteria": List[Criterion],
    "unsatisfied_criteria": List[Criterion],
    "missing_information": List[str],
    "reasoning": str
}
```

**Logic Flow**:
1. Parse scheme eligibility criteria
2. Apply rule-based validation for explicit criteria
3. Use LLM for complex/ambiguous criteria
4. Compute overall eligibility score
5. Generate explainable reasoning

**Implementation**:
```python
class EligibilityValidatorAgent:
    def __init__(
        self,
        validation_service: ValidationService,
        llm_service: LLMService
    ):
        self.validation_service = validation_service
        self.llm_service = llm_service
        
    async def execute(self, state: AgentState) -> AgentState:
        scheme = state.current_scheme
        context = state.user_context
        
        # Parse criteria
        criteria = self._parse_criteria(scheme.eligibility_criteria)
        
        # Validate each criterion
        results = []
        for criterion in criteria:
            if criterion.type == "rule_based":
                result = self._validate_rule_based(criterion, context)
            else:
                result = await self._validate_llm_based(criterion, context)
            results.append(result)
        
        # Compute overall eligibility
        eligibility = self._compute_eligibility(results)
        
        # Generate reasoning
        reasoning = await self._generate_reasoning(results, context, scheme)
        
        state.eligibility_result = eligibility
        state.eligibility_reasoning = reasoning
        
        return state
        
    def _validate_rule_based(
        self,
        criterion: Criterion,
        context: UserContext
    ) -> ValidationResult:
        """Rule-based validation"""
        if criterion.field == "age":
            return self.validation_service.validate_age(
                context.age,
                criterion.constraints
            )
        elif criterion.field == "income":
            return self.validation_service.validate_income(
                context.income,
                criterion.constraints
            )
        # ... other fields
        
    async def _validate_llm_based(
        self,
        criterion: Criterion,
        context: UserContext
    ) -> ValidationResult:
        """LLM-based validation for complex criteria"""
        prompt = f"""
        Evaluate if the user satisfies this eligibility criterion:
        
        Criterion: {criterion.description}
        
        User Context:
        {context.to_dict()}
        
        Provide:
        1. satisfied: true/false
        2. confidence: 0-1
        3. reasoning: explanation
        
        Output JSON format.
        """
        
        response = await self.llm_service.generate_structured(
            prompt=prompt,
            schema=VALIDATION_SCHEMA
        )
        
        return ValidationResult(**response)
```

#### 4.2.4 Documentation Agent

**Purpose**: Process, validate, and verify uploaded documents

**Input**:
```python
{
    "documents": List[UploadedDocument],
    "required_documents": List[str],
    "user_context": UserContext
}
```

**Output**:
```python
{
    "processed_documents": List[ProcessedDocument],
    "validation_results": List[DocumentValidation],
    "missing_documents": List[str],
    "issues": List[DocumentIssue]
}
```

**Logic Flow**:
1. Perform OCR on uploaded documents
2. Classify document types
3. Extract structured fields
4. Validate against user context
5. Check for missing documents
6. Generate validation report

**Implementation**:
```python
class DocumentationAgent:
    def __init__(
        self,
        ocr_service: OCRService,
        validation_service: ValidationService,
        llm_service: LLMService
    ):
        self.ocr_service = ocr_service
        self.validation_service = validation_service
        self.llm_service = llm_service
        
    async def execute(self, state: AgentState) -> AgentState:
        processed_docs = []
        
        for doc in state.uploaded_documents:
            # OCR
            ocr_result = await self.ocr_service.extract_text(
                doc.image_bytes,
                language="eng+hin"
            )
            
            # Classify
            doc_type = await self.ocr_service.classify_document(
                ocr_result.text,
                doc.image_bytes
            )
            
            # Extract fields
            fields = await self.ocr_service.extract_fields(
                ocr_result.text,
                doc_type
            )
            
            # Validate
            validation = await self._validate_document(
                fields,
                doc_type,
                state.user_context
            )
            
            processed_docs.append(ProcessedDocument(
                original=doc,
                type=doc_type,
                extracted_fields=fields,
                validation=validation
            ))
        
        # Check missing documents
        missing = self._check_missing_documents(
            processed_docs,
            state.required_documents
        )
        
        state.processed_documents = processed_docs
        state.missing_documents = missing
        
        return state
        
    async def _validate_document(
        self,
        fields: dict,
        doc_type: DocumentType,
        context: UserContext
    ) -> DocumentValidation:
        """Validate document fields against user context"""
        issues = []
        
        # Name matching
        if "name" in fields:
            if not self._fuzzy_match(fields["name"], context.name):
                issues.append("Name mismatch")
        
        # Date of birth matching
        if "dob" in fields:
            if fields["dob"] != context.date_of_birth:
                issues.append("Date of birth mismatch")
        
        # Expiry check
        if "expiry_date" in fields:
            if self._is_expired(fields["expiry_date"]):
                issues.append("Document expired")
        
        return DocumentValidation(
            is_valid=len(issues) == 0,
            issues=issues,
            confidence=self._compute_confidence(fields, issues)
        )
```

#### 4.2.5 Application Readiness Agent

**Purpose**: Generate structured application-ready outputs

**Input**:
```python
{
    "user_context": UserContext,
    "scheme": Scheme,
    "processed_documents": List[ProcessedDocument],
    "eligibility_result": EligibilityResult
}
```

**Output**:
```python
{
    "application_json": dict,
    "prefilled_form": dict,
    "submission_guide": str,
    "checklist": List[str]
}
```

**Logic Flow**:
1. Map user context to scheme application fields
2. Extract document data for application fields
3. Generate structured JSON
4. Create submission checklist
5. Generate step-by-step guide

**Implementation**:
```python
class ApplicationReadinessAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
    async def execute(self, state: AgentState) -> AgentState:
        scheme = state.current_scheme
        context = state.user_context
        docs = state.processed_documents
        
        # Map fields
        application_data = self._map_fields(
            scheme.application_fields,
            context,
            docs
        )
        
        # Generate checklist
        checklist = self._generate_checklist(
            scheme,
            docs,
            state.missing_documents
        )
        
        # Generate submission guide
        guide = await self._generate_submission_guide(
            scheme,
            application_data,
            checklist
        )
        
        state.application_json = application_data
        state.submission_checklist = checklist
        state.submission_guide = guide
        
        return state
        
    def _map_fields(
        self,
        required_fields: List[str],
        context: UserContext,
        docs: List[ProcessedDocument]
    ) -> dict:
        """Map user context and document data to application fields"""
        application_data = {}
        
        # Map from user context
        field_mapping = {
            "name": context.name,
            "age": context.age,
            "gender": context.gender,
            "address": context.address,
            "mobile": context.mobile,
            "income": context.income,
            # ... more mappings
        }
        
        # Map from documents
        for doc in docs:
            if doc.type == "aadhaar":
                application_data["aadhaar_number"] = doc.extracted_fields.get("aadhaar_number")
            elif doc.type == "pan":
                application_data["pan_number"] = doc.extracted_fields.get("pan_number")
            # ... more document types
        
        return application_data
```

### 4.3 Agent Orchestration Flow

```python
from langgraph.graph import StateGraph, END

# Define agent state
class AgentState(TypedDict):
    user_message: str
    conversation_history: List[Message]
    user_context: UserContext
    retrieved_schemes: List[Scheme]
    current_scheme: Scheme
    eligibility_result: EligibilityResult
    uploaded_documents: List[UploadedDocument]
    processed_documents: List[ProcessedDocument]
    application_json: dict
    next_action: str

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("context_builder", context_builder_agent.execute)
workflow.add_node("scheme_researcher", scheme_researcher_agent.execute)
workflow.add_node("eligibility_validator", eligibility_validator_agent.execute)
workflow.add_node("documentation", documentation_agent.execute)
workflow.add_node("app_readiness", app_readiness_agent.execute)

# Define edges
workflow.set_entry_point("context_builder")

workflow.add_conditional_edges(
    "context_builder",
    lambda state: "scheme_researcher" if state.user_context.is_complete() else "context_builder",
    {
        "context_builder": "context_builder",
        "scheme_researcher": "scheme_researcher"
    }
)

workflow.add_edge("scheme_researcher", "eligibility_validator")

workflow.add_conditional_edges(
    "eligibility_validator",
    lambda state: "documentation" if state.eligibility_result.is_eligible else END,
    {
        "documentation": "documentation",
        END: END
    }
)

workflow.add_edge("documentation", "app_readiness")
workflow.add_edge("app_readiness", END)

# Compile
app = workflow.compile()
```


## 5. Data Flow Design

### 5.1 End-to-End Data Flow

```
User Input → API Gateway → Session Manager → Agent Orchestrator
                                                      ↓
                                            Context Builder Agent
                                                      ↓
                                            [User Context Updated]
                                                      ↓
                                            Scheme Researcher Agent
                                                      ↓
                                            [RAG Service Query]
                                                      ↓
                                            [Vector DB Retrieval]
                                                      ↓
                                            [Schemes Retrieved]
                                                      ↓
                                            Eligibility Validator Agent
                                                      ↓
                                            [Rule-based + LLM Validation]
                                                      ↓
                                            [Eligibility Determined]
                                                      ↓
                                            Documentation Agent
                                                      ↓
                                            [OCR Processing]
                                                      ↓
                                            [Document Validation]
                                                      ↓
                                            Application Readiness Agent
                                                      ↓
                                            [Application JSON Generated]
                                                      ↓
                                            Response Formatter
                                                      ↓
                                            API Gateway → User
```

### 5.2 Conversation Flow Example

**User**: "I am a farmer from Maharashtra, I have 2 acres of land"

**Step 1: Context Building**
```json
{
  "extracted_attributes": {
    "occupation": "farmer",
    "location": "Maharashtra",
    "land_ownership": 2.0
  },
  "confidence_scores": {
    "occupation": 0.95,
    "location": 0.95,
    "land_ownership": 0.90
  },
  "missing_critical": ["age", "income", "caste_category"]
}
```

**System Response**: "I understand you're a farmer in Maharashtra with 2 acres of land. To find the best schemes for you, could you tell me your age and approximate annual income?"

**User**: "I am 45 years old and my income is around 80,000 per year"

**Step 2: Context Update + Scheme Research**
```json
{
  "user_context": {
    "occupation": "farmer",
    "location": "Maharashtra",
    "land_ownership": 2.0,
    "age": 45,
    "income": 80000
  },
  "retrieved_schemes": [
    {
      "name": "PM-KISAN",
      "relevance_score": 0.92
    },
    {
      "name": "Maharashtra Farmer Loan Waiver Scheme",
      "relevance_score": 0.88
    }
  ]
}
```

**Step 3: Eligibility Validation**
```json
{
  "scheme": "PM-KISAN",
  "is_eligible": true,
  "confidence": 0.85,
  "satisfied_criteria": [
    "Is a farmer",
    "Owns cultivable land",
    "Land ownership < 2 hectares"
  ],
  "unsatisfied_criteria": [],
  "missing_information": ["bank_account_details"]
}
```

**System Response**: "Great news! You're eligible for PM-KISAN scheme. You can receive ₹6,000 per year. Please upload your Aadhaar card and bank passbook to proceed."

### 5.3 Document Processing Flow

```
Document Upload → Object Storage → OCR Service
                                        ↓
                                  [Text Extraction]
                                        ↓
                                  [Document Classification]
                                        ↓
                                  [Field Extraction]
                                        ↓
                                  [Validation Service]
                                        ↓
                                  [Match with User Context]
                                        ↓
                                  [Validation Result]
                                        ↓
                                  [Store in Database]
```

### 5.4 RAG Pipeline Flow

```
User Query → Query Embedding → Vector Search
                                     ↓
                              [Top-K Retrieval]
                                     ↓
                              Keyword Search
                                     ↓
                              [Hybrid Merge]
                                     ↓
                              Context Reranking
                                     ↓
                              [Top-5 Schemes]
                                     ↓
                              LLM Augmentation
                                     ↓
                              [Enriched Response]
```


## 6. Data Models and Schema Definitions

### 6.1 User Context Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class UserContext(BaseModel):
    user_id: str
    session_id: str
    
    # Demographics
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, pattern="^(Male|Female|Other)$")
    date_of_birth: Optional[date] = None
    
    # Location
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    pincode: Optional[str] = Field(None, pattern="^[0-9]{6}$")
    
    # Economic
    occupation: Optional[str] = None
    annual_income: Optional[float] = Field(None, ge=0)
    bpl_status: Optional[bool] = None
    
    # Social
    caste_category: Optional[str] = Field(None, pattern="^(General|OBC|SC|ST)$")
    religion: Optional[str] = None
    minority_status: Optional[bool] = None
    
    # Family
    marital_status: Optional[str] = None
    household_size: Optional[int] = Field(None, ge=1)
    dependents: Optional[int] = Field(None, ge=0)
    
    # Special Categories
    disability: Optional[bool] = None
    disability_percentage: Optional[int] = Field(None, ge=0, le=100)
    widow: Optional[bool] = None
    
    # Assets
    land_ownership: Optional[float] = Field(None, ge=0)  # in acres
    owns_house: Optional[bool] = None
    
    # Contact
    mobile: Optional[str] = Field(None, pattern="^[0-9]{10}$")
    email: Optional[str] = None
    
    # Confidence Scores
    confidence_scores: dict[str, float] = {}
    
    # Metadata
    language_preference: str = "en"
    created_at: date = Field(default_factory=date.today)
    updated_at: date = Field(default_factory=date.today)
    
    def is_complete(self, required_fields: List[str]) -> bool:
        """Check if required fields are present"""
        for field in required_fields:
            if getattr(self, field) is None:
                return False
        return True
```

### 6.2 Scheme Schema

```python
class EligibilityCriterion(BaseModel):
    field: str
    operator: str  # "eq", "gt", "lt", "gte", "lte", "in", "contains"
    value: any
    description: str
    is_mandatory: bool = True
    validation_type: str = "rule_based"  # or "llm_based"

class Scheme(BaseModel):
    scheme_id: str
    name: str
    name_local: dict[str, str]  # {"hi": "...", "mr": "..."}
    description: str
    description_local: dict[str, str]
    
    # Classification
    category: str  # "agriculture", "education", "health", etc.
    subcategory: Optional[str] = None
    scheme_type: str  # "central", "state"
    state: Optional[str] = None  # for state schemes
    
    # Benefits
    benefit_type: str  # "financial", "subsidy", "service"
    benefit_amount: Optional[float] = None
    benefit_description: str
    
    # Eligibility
    eligibility_criteria: List[EligibilityCriterion]
    eligibility_summary: str
    
    # Documents
    required_documents: List[str]
    optional_documents: List[str] = []
    
    # Application
    application_process: str
    application_url: Optional[str] = None
    application_deadline: Optional[date] = None
    
    # Metadata
    ministry: Optional[str] = None
    department: Optional[str] = None
    official_website: Optional[str] = None
    helpline: Optional[str] = None
    
    # Vector Embedding
    embedding: Optional[List[float]] = None
    
    # Timestamps
    created_at: date
    updated_at: date
    last_verified: date
```

### 6.3 Document Schema

```python
class UploadedDocument(BaseModel):
    document_id: str
    user_id: str
    session_id: str
    filename: str
    file_size: int
    mime_type: str
    storage_path: str
    uploaded_at: date

class ProcessedDocument(BaseModel):
    document_id: str
    document_type: str  # "aadhaar", "pan", "income_certificate", etc.
    classification_confidence: float
    
    # OCR Results
    extracted_text: str
    ocr_confidence: float
    
    # Extracted Fields
    extracted_fields: dict
    
    # Validation
    is_valid: bool
    validation_issues: List[str]
    validation_confidence: float
    
    # Metadata
    processed_at: date

class DocumentValidation(BaseModel):
    document_id: str
    is_valid: bool
    confidence: float
    issues: List[str]
    matched_fields: dict[str, bool]
```

### 6.4 Eligibility Result Schema

```python
class CriterionResult(BaseModel):
    criterion: EligibilityCriterion
    satisfied: bool
    confidence: float
    reasoning: str
    user_value: any

class EligibilityResult(BaseModel):
    scheme_id: str
    user_id: str
    
    # Overall Result
    is_eligible: bool
    confidence: float
    eligibility_score: float  # 0-1
    
    # Detailed Results
    satisfied_criteria: List[CriterionResult]
    unsatisfied_criteria: List[CriterionResult]
    missing_information: List[str]
    
    # Reasoning
    reasoning: str
    reasoning_steps: List[str]
    
    # Metadata
    evaluated_at: date
    evaluation_method: str  # "rule_based", "llm_based", "hybrid"
```

### 6.5 Application Readiness Schema

```python
class ApplicationReadiness(BaseModel):
    user_id: str
    scheme_id: str
    
    # Application Data
    application_json: dict
    prefilled_form: dict
    
    # Readiness Status
    is_ready: bool
    completeness_score: float  # 0-1
    
    # Checklist
    completed_items: List[str]
    pending_items: List[str]
    
    # Documents
    validated_documents: List[str]
    missing_documents: List[str]
    
    # Submission Guide
    submission_steps: List[str]
    portal_url: Optional[str]
    estimated_time: Optional[str]
    
    # Metadata
    generated_at: date
```

### 6.6 Conversation Schema

```python
class Message(BaseModel):
    message_id: str
    session_id: str
    role: str  # "user", "assistant", "system"
    content: str
    content_type: str  # "text", "voice", "document"
    
    # Metadata
    language: str
    timestamp: date
    
    # Agent Metadata (for assistant messages)
    agent_name: Optional[str] = None
    confidence: Optional[float] = None

class Session(BaseModel):
    session_id: str
    user_id: str
    
    # State
    current_state: str  # "context_building", "scheme_research", etc.
    user_context: UserContext
    
    # Conversation
    messages: List[Message]
    
    # Metadata
    started_at: date
    last_activity: date
    is_active: bool
```

### 6.7 Database Schema (PostgreSQL)

```sql
-- Users Table
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    language_preference VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Contexts Table
CREATE TABLE user_contexts (
    context_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    session_id UUID,
    
    -- Demographics
    age INTEGER,
    gender VARCHAR(20),
    date_of_birth DATE,
    
    -- Location
    state VARCHAR(100),
    district VARCHAR(100),
    village VARCHAR(100),
    pincode VARCHAR(6),
    
    -- Economic
    occupation VARCHAR(100),
    annual_income DECIMAL(12, 2),
    bpl_status BOOLEAN,
    
    -- Social
    caste_category VARCHAR(20),
    minority_status BOOLEAN,
    
    -- Family
    household_size INTEGER,
    dependents INTEGER,
    
    -- Special
    disability BOOLEAN,
    disability_percentage INTEGER,
    
    -- Assets
    land_ownership DECIMAL(10, 2),
    
    -- Contact
    mobile VARCHAR(10),
    email VARCHAR(255),
    
    -- Metadata
    confidence_scores JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schemes Table
CREATE TABLE schemes (
    scheme_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    name_local JSONB,
    description TEXT,
    description_local JSONB,
    
    -- Classification
    category VARCHAR(100),
    subcategory VARCHAR(100),
    scheme_type VARCHAR(20),
    state VARCHAR(100),
    
    -- Benefits
    benefit_type VARCHAR(50),
    benefit_amount DECIMAL(12, 2),
    benefit_description TEXT,
    
    -- Eligibility
    eligibility_criteria JSONB NOT NULL,
    eligibility_summary TEXT,
    
    -- Documents
    required_documents JSONB,
    optional_documents JSONB,
    
    -- Application
    application_process TEXT,
    application_url VARCHAR(500),
    application_deadline DATE,
    
    -- Metadata
    ministry VARCHAR(200),
    department VARCHAR(200),
    official_website VARCHAR(500),
    helpline VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_verified TIMESTAMP
);

-- Documents Table
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    session_id UUID,
    
    -- File Info
    filename VARCHAR(255),
    file_size INTEGER,
    mime_type VARCHAR(100),
    storage_path VARCHAR(500),
    
    -- Processing
    document_type VARCHAR(50),
    classification_confidence DECIMAL(3, 2),
    extracted_text TEXT,
    ocr_confidence DECIMAL(3, 2),
    extracted_fields JSONB,
    
    -- Validation
    is_valid BOOLEAN,
    validation_issues JSONB,
    validation_confidence DECIMAL(3, 2),
    
    -- Timestamps
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Eligibility Results Table
CREATE TABLE eligibility_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    scheme_id UUID REFERENCES schemes(scheme_id),
    
    -- Results
    is_eligible BOOLEAN,
    confidence DECIMAL(3, 2),
    eligibility_score DECIMAL(3, 2),
    
    -- Details
    satisfied_criteria JSONB,
    unsatisfied_criteria JSONB,
    missing_information JSONB,
    
    -- Reasoning
    reasoning TEXT,
    reasoning_steps JSONB,
    evaluation_method VARCHAR(50),
    
    -- Timestamp
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions Table
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id),
    
    -- State
    current_state VARCHAR(50),
    
    -- Metadata
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP
);

-- Messages Table
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(session_id),
    
    -- Content
    role VARCHAR(20),
    content TEXT,
    content_type VARCHAR(20),
    language VARCHAR(10),
    
    -- Agent Metadata
    agent_name VARCHAR(100),
    confidence DECIMAL(3, 2),
    
    -- Timestamp
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_user_contexts_user_id ON user_contexts(user_id);
CREATE INDEX idx_schemes_category ON schemes(category);
CREATE INDEX idx_schemes_state ON schemes(state);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_eligibility_results_user_id ON eligibility_results(user_id);
CREATE INDEX idx_eligibility_results_scheme_id ON eligibility_results(scheme_id);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);
```


## 7. API Design and Endpoints

### 7.1 Authentication APIs

#### POST /api/v1/auth/register
```json
Request:
{
  "phone_number": "9876543210",
  "language_preference": "hi"
}

Response:
{
  "user_id": "uuid",
  "otp_sent": true,
  "message": "OTP sent to your phone"
}
```

#### POST /api/v1/auth/verify-otp
```json
Request:
{
  "phone_number": "9876543210",
  "otp": "123456"
}

Response:
{
  "access_token": "jwt_token",
  "refresh_token": "refresh_token",
  "user_id": "uuid",
  "expires_in": 3600
}
```

#### POST /api/v1/auth/refresh
```json
Request:
{
  "refresh_token": "refresh_token"
}

Response:
{
  "access_token": "new_jwt_token",
  "expires_in": 3600
}
```

### 7.2 Session APIs

#### POST /api/v1/sessions
```json
Request:
{
  "user_id": "uuid",
  "language": "hi"
}

Response:
{
  "session_id": "uuid",
  "started_at": "2026-02-15T12:00:00Z",
  "expires_at": "2026-02-15T12:30:00Z"
}
```

#### GET /api/v1/sessions/{session_id}
```json
Response:
{
  "session_id": "uuid",
  "user_id": "uuid",
  "current_state": "context_building",
  "user_context": {...},
  "messages": [...],
  "is_active": true
}
```

### 7.3 Chat APIs

#### POST /api/v1/chat/message
```json
Request:
{
  "session_id": "uuid",
  "message": "I am a farmer from Maharashtra",
  "message_type": "text",
  "language": "en"
}

Response:
{
  "message_id": "uuid",
  "response": "I understand you're a farmer in Maharashtra...",
  "response_type": "text",
  "agent_name": "context_builder",
  "confidence": 0.95,
  "next_action": "clarification_needed",
  "suggestions": [
    "Tell me your age",
    "Upload documents"
  ]
}
```

#### POST /api/v1/chat/voice
```json
Request (multipart/form-data):
{
  "session_id": "uuid",
  "audio_file": <binary>,
  "language": "hi"
}

Response:
{
  "transcript": "मैं महाराष्ट्र का किसान हूं",
  "response": "मैं समझता हूं कि आप महाराष्ट्र के किसान हैं...",
  "audio_response": <binary>,
  "confidence": 0.92
}
```

### 7.4 Scheme APIs

#### GET /api/v1/schemes/search
```json
Query Parameters:
- query: string
- state: string (optional)
- category: string (optional)
- limit: integer (default: 10)

Response:
{
  "schemes": [
    {
      "scheme_id": "uuid",
      "name": "PM-KISAN",
      "description": "...",
      "benefit_amount": 6000,
      "relevance_score": 0.92
    }
  ],
  "total": 25,
  "page": 1
}
```

#### GET /api/v1/schemes/{scheme_id}
```json
Response:
{
  "scheme_id": "uuid",
  "name": "PM-KISAN",
  "name_local": {"hi": "पीएम-किसान"},
  "description": "...",
  "category": "agriculture",
  "benefit_type": "financial",
  "benefit_amount": 6000,
  "eligibility_criteria": [...],
  "required_documents": [...],
  "application_url": "https://..."
}
```

#### POST /api/v1/schemes/{scheme_id}/check-eligibility
```json
Request:
{
  "session_id": "uuid",
  "user_context": {...}
}

Response:
{
  "is_eligible": true,
  "confidence": 0.88,
  "eligibility_score": 0.85,
  "satisfied_criteria": [
    {
      "criterion": "Is a farmer",
      "satisfied": true,
      "confidence": 0.95,
      "reasoning": "User stated occupation as farmer"
    }
  ],
  "unsatisfied_criteria": [],
  "missing_information": ["bank_account_details"],
  "reasoning": "You satisfy all eligibility criteria..."
}
```

### 7.5 Document APIs

#### POST /api/v1/documents/upload
```json
Request (multipart/form-data):
{
  "session_id": "uuid",
  "files": [<binary>, <binary>],
  "document_types": ["aadhaar", "income_certificate"]
}

Response:
{
  "uploaded_documents": [
    {
      "document_id": "uuid",
      "filename": "aadhaar.jpg",
      "status": "processing"
    }
  ]
}
```

#### GET /api/v1/documents/{document_id}/status
```json
Response:
{
  "document_id": "uuid",
  "status": "processed",
  "document_type": "aadhaar",
  "classification_confidence": 0.95,
  "extracted_fields": {
    "name": "John Doe",
    "aadhaar_number": "XXXX-XXXX-1234",
    "dob": "1980-01-01"
  },
  "is_valid": true,
  "validation_issues": []
}
```

#### POST /api/v1/documents/validate
```json
Request:
{
  "session_id": "uuid",
  "document_ids": ["uuid1", "uuid2"],
  "scheme_id": "uuid"
}

Response:
{
  "validation_results": [
    {
      "document_id": "uuid1",
      "document_type": "aadhaar",
      "is_valid": true,
      "issues": [],
      "matched_fields": {
        "name": true,
        "dob": true
      }
    }
  ],
  "missing_documents": ["income_certificate"],
  "overall_readiness": 0.75
}
```

### 7.6 Application Readiness APIs

#### POST /api/v1/applications/generate
```json
Request:
{
  "session_id": "uuid",
  "scheme_id": "uuid"
}

Response:
{
  "application_id": "uuid",
  "is_ready": true,
  "completeness_score": 0.95,
  "application_json": {
    "name": "John Doe",
    "age": 45,
    "occupation": "farmer",
    ...
  },
  "submission_checklist": [
    "✓ Aadhaar card uploaded",
    "✓ Income certificate uploaded",
    "✗ Bank passbook pending"
  ],
  "submission_guide": "Step 1: Visit https://...",
  "portal_url": "https://pmkisan.gov.in"
}
```

#### GET /api/v1/applications/{application_id}/download
```json
Query Parameters:
- format: "json" | "pdf"

Response:
- JSON: application data
- PDF: prefilled form (binary)
```

### 7.7 User Context APIs

#### GET /api/v1/users/{user_id}/context
```json
Response:
{
  "user_id": "uuid",
  "context": {
    "age": 45,
    "occupation": "farmer",
    "location": "Maharashtra",
    ...
  },
  "confidence_scores": {
    "age": 0.95,
    "occupation": 0.95,
    ...
  },
  "completeness": 0.75
}
```

#### PUT /api/v1/users/{user_id}/context
```json
Request:
{
  "updates": {
    "annual_income": 80000,
    "caste_category": "OBC"
  }
}

Response:
{
  "updated_context": {...},
  "updated_at": "2026-02-15T12:00:00Z"
}
```

### 7.8 Analytics APIs (Admin)

#### GET /api/v1/analytics/schemes/popular
```json
Response:
{
  "schemes": [
    {
      "scheme_id": "uuid",
      "name": "PM-KISAN",
      "query_count": 1250,
      "eligibility_check_count": 850,
      "application_generated_count": 420
    }
  ]
}
```

#### GET /api/v1/analytics/users/demographics
```json
Response:
{
  "total_users": 5000,
  "by_state": {
    "Maharashtra": 1200,
    "Karnataka": 800,
    ...
  },
  "by_occupation": {
    "farmer": 2500,
    "laborer": 1000,
    ...
  }
}
```

### 7.9 Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid phone number format",
    "details": {
      "field": "phone_number",
      "constraint": "must be 10 digits"
    },
    "request_id": "uuid"
  }
}
```

### 7.10 API Rate Limits

- Authentication endpoints: 5 requests/minute per IP
- Chat endpoints: 30 requests/minute per user
- Document upload: 10 requests/minute per user
- Scheme search: 60 requests/minute per user


## 8. RAG Pipeline Design

### 8.1 RAG Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCHEME KNOWLEDGE BASE                     │
│  (Government scheme documents, PDFs, web pages)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DOCUMENT PROCESSING                        │
│  • PDF extraction                                            │
│  • HTML parsing                                              │
│  • Text cleaning                                             │
│  • Metadata extraction                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      CHUNKING STRATEGY                       │
│  • Chunk size: 512 tokens                                    │
│  • Overlap: 50 tokens                                        │
│  • Semantic chunking (preserve context)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION                      │
│  • Model: text-embedding-3-large (OpenAI)                    │
│  • Dimension: 1536                                           │
│  • Batch processing                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     VECTOR DATABASE                          │
│  • Pinecone / Weaviate / Chroma                              │
│  • Cosine similarity search                                  │
│  • Metadata filtering                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                        │
│  User Query → Embedding → Vector Search → Reranking         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   CONTEXT AUGMENTATION                       │
│  Retrieved Chunks + User Context → LLM Prompt                │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Document Processing Pipeline

```python
class SchemeDocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def process_scheme_document(self, document: dict) -> List[Chunk]:
        """Process scheme document into chunks"""
        
        # Extract text
        text = self._extract_text(document)
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract metadata
        metadata = self._extract_metadata(document)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create chunk objects with metadata
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                chunk_id=f"{document['scheme_id']}_chunk_{i}",
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _extract_metadata(self, document: dict) -> dict:
        """Extract scheme metadata"""
        return {
            "scheme_id": document["scheme_id"],
            "scheme_name": document["name"],
            "category": document["category"],
            "state": document.get("state"),
            "scheme_type": document["scheme_type"],
            "benefit_amount": document.get("benefit_amount"),
            "ministry": document.get("ministry")
        }
```

### 8.3 Embedding Generation

```python
class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self.client = OpenAI()
        
    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
```

### 8.4 Vector Database Setup

```python
# Using Pinecone
import pinecone

class VectorDBService:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        
        # Create index if not exists
        if "schemes" not in pinecone.list_indexes():
            pinecone.create_index(
                name="schemes",
                dimension=1536,
                metric="cosine",
                metadata_config={
                    "indexed": [
                        "scheme_type",
                        "category",
                        "state",
                        "benefit_amount"
                    ]
                }
            )
        
        self.index = pinecone.Index("schemes")
    
    async def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Insert/update chunks in vector database"""
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.text,
                    **chunk.metadata
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: dict = None
    ) -> List[dict]:
        """Search for similar chunks"""
        
        # Build filter
        filter_dict = {}
        if filters:
            if filters.get("state"):
                filter_dict["state"] = {"$eq": filters["state"]}
            if filters.get("category"):
                filter_dict["category"] = {"$eq": filters["category"]}
            if filters.get("scheme_type"):
                filter_dict["scheme_type"] = {"$eq": filters["scheme_type"]}
        
        # Query
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        return results.matches
```

### 8.5 Hybrid Search Implementation

```python
class HybridSearchService:
    def __init__(
        self,
        vector_db: VectorDBService,
        embedding_service: EmbeddingService
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        
    async def hybrid_search(
        self,
        query: str,
        user_context: UserContext,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Document]:
        """Perform hybrid search combining semantic and keyword search"""
        
        # Semantic search
        query_embedding = await self.embedding_service.generate_embeddings([query])
        
        filters = self._build_filters(user_context)
        
        semantic_results = await self.vector_db.search(
            query_embedding=query_embedding[0],
            top_k=top_k * 2,  # Get more for reranking
            filters=filters
        )
        
        # Keyword search (using metadata)
        keyword_results = await self._keyword_search(query, user_context)
        
        # Merge and rerank
        merged_results = self._merge_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Deduplicate by scheme_id
        unique_schemes = self._deduplicate_by_scheme(merged_results)
        
        return unique_schemes[:top_k]
    
    def _build_filters(self, context: UserContext) -> dict:
        """Build filters from user context"""
        filters = {}
        
        if context.state:
            filters["state"] = context.state
        
        # Add central schemes
        filters["scheme_type"] = {"$in": ["central", context.state]}
        
        return filters
    
    async def _keyword_search(
        self,
        query: str,
        context: UserContext
    ) -> List[dict]:
        """Keyword-based search using PostgreSQL full-text search"""
        # Implementation using PostgreSQL
        pass
    
    def _merge_results(
        self,
        semantic_results: List[dict],
        keyword_results: List[dict],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[dict]:
        """Merge and rerank results"""
        
        # Create score map
        score_map = {}
        
        # Add semantic scores
        for i, result in enumerate(semantic_results):
            scheme_id = result["metadata"]["scheme_id"]
            score = result["score"] * semantic_weight
            score_map[scheme_id] = score_map.get(scheme_id, 0) + score
        
        # Add keyword scores
        for i, result in enumerate(keyword_results):
            scheme_id = result["scheme_id"]
            score = result["score"] * keyword_weight
            score_map[scheme_id] = score_map.get(scheme_id, 0) + score
        
        # Sort by combined score
        sorted_schemes = sorted(
            score_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_schemes
```

### 8.6 Context-Aware Reranking

```python
class ContextReranker:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
    async def rerank(
        self,
        query: str,
        user_context: UserContext,
        schemes: List[Scheme]
    ) -> List[Scheme]:
        """Rerank schemes based on user context"""
        
        prompt = f"""
        Given the user context and query, rank the following schemes by relevance.
        
        User Context:
        - Age: {user_context.age}
        - Occupation: {user_context.occupation}
        - Location: {user_context.state}
        - Income: {user_context.annual_income}
        - Category: {user_context.caste_category}
        
        Query: {query}
        
        Schemes:
        {self._format_schemes(schemes)}
        
        Rank the schemes from most to least relevant.
        Output JSON array of scheme_ids in ranked order.
        """
        
        response = await self.llm_service.generate_structured(
            prompt=prompt,
            schema={"type": "array", "items": {"type": "string"}}
        )
        
        # Reorder schemes based on ranking
        ranked_schemes = []
        for scheme_id in response:
            scheme = next((s for s in schemes if s.scheme_id == scheme_id), None)
            if scheme:
                ranked_schemes.append(scheme)
        
        return ranked_schemes
```

### 8.7 RAG Prompt Template

```python
RAG_PROMPT_TEMPLATE = """
You are a government scheme expert helping rural citizens find relevant welfare schemes.

User Context:
{user_context}

Retrieved Scheme Information:
{retrieved_chunks}

User Query: {query}

Based on the user's context and the retrieved scheme information, provide:
1. A list of relevant schemes
2. Brief explanation of each scheme's benefits
3. Preliminary eligibility assessment
4. Next steps for the user

Be conversational, empathetic, and use simple language.
If speaking in Hindi or regional language, use appropriate translations.

Response:
"""
```

### 8.8 Knowledge Base Update Pipeline

```python
class KnowledgeBaseUpdater:
    def __init__(
        self,
        document_processor: SchemeDocumentProcessor,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService
    ):
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        
    async def update_scheme(self, scheme_document: dict):
        """Update or add scheme to knowledge base"""
        
        # Process document
        chunks = self.document_processor.process_scheme_document(scheme_document)
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)
        
        # Upsert to vector database
        await self.vector_db.upsert_chunks(chunks, embeddings)
        
        # Update PostgreSQL
        await self._update_postgres(scheme_document)
    
    async def bulk_update(self, scheme_documents: List[dict]):
        """Bulk update schemes"""
        for doc in scheme_documents:
            await self.update_scheme(doc)
```


## 9. Agent Orchestration Design

### 9.1 LangGraph State Machine

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    # Input
    user_message: str
    message_type: str  # "text", "voice", "document"
    language: str
    
    # Session
    session_id: str
    user_id: str
    conversation_history: Sequence[Message]
    
    # User Context
    user_context: UserContext
    context_confidence: float
    missing_attributes: list[str]
    
    # Scheme Research
    search_query: str
    retrieved_schemes: list[Scheme]
    retrieval_reasoning: str
    
    # Eligibility
    current_scheme: Scheme
    eligibility_result: EligibilityResult
    eligibility_reasoning: str
    
    # Documentation
    uploaded_documents: list[UploadedDocument]
    processed_documents: list[ProcessedDocument]
    missing_documents: list[str]
    
    # Application
    application_json: dict
    submission_checklist: list[str]
    submission_guide: str
    
    # Control Flow
    next_action: str
    error: str
    
    # Response
    response_message: str
    response_data: dict

def create_agent_workflow():
    """Create LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("scheme_researcher", scheme_researcher_node)
    workflow.add_node("eligibility_validator", eligibility_validator_node)
    workflow.add_node("documentation_processor", documentation_processor_node)
    workflow.add_node("application_generator", application_generator_node)
    workflow.add_node("response_formatter", response_formatter_node)
    
    # Set entry point
    workflow.set_entry_point("context_builder")
    
    # Define conditional edges
    workflow.add_conditional_edges(
        "context_builder",
        route_after_context_building,
        {
            "scheme_researcher": "scheme_researcher",
            "clarification": "response_formatter",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "scheme_researcher",
        route_after_scheme_research,
        {
            "eligibility_validator": "eligibility_validator",
            "no_schemes_found": "response_formatter",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "eligibility_validator",
        route_after_eligibility,
        {
            "documentation_processor": "documentation_processor",
            "not_eligible": "response_formatter",
            "need_more_info": "response_formatter",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "documentation_processor",
        route_after_documentation,
        {
            "application_generator": "application_generator",
            "missing_documents": "response_formatter",
            "error": END
        }
    )
    
    workflow.add_edge("application_generator", "response_formatter")
    workflow.add_edge("response_formatter", END)
    
    return workflow.compile()

# Routing functions
def route_after_context_building(state: AgentState) -> str:
    """Determine next step after context building"""
    
    if state.get("error"):
        return "error"
    
    # Check if we have enough context to search schemes
    required_fields = ["occupation", "state", "age"]
    missing = [f for f in required_fields if not getattr(state["user_context"], f)]
    
    if missing:
        state["next_action"] = "clarification"
        return "clarification"
    
    return "scheme_researcher"

def route_after_scheme_research(state: AgentState) -> str:
    """Determine next step after scheme research"""
    
    if state.get("error"):
        return "error"
    
    if not state["retrieved_schemes"]:
        state["next_action"] = "no_schemes_found"
        return "no_schemes_found"
    
    # Automatically validate eligibility for top scheme
    state["current_scheme"] = state["retrieved_schemes"][0]
    return "eligibility_validator"

def route_after_eligibility(state: AgentState) -> str:
    """Determine next step after eligibility validation"""
    
    if state.get("error"):
        return "error"
    
    result = state["eligibility_result"]
    
    if not result.is_eligible:
        state["next_action"] = "not_eligible"
        return "not_eligible"
    
    if result.missing_information:
        state["next_action"] = "need_more_info"
        return "need_more_info"
    
    # Check if documents are uploaded
    if not state.get("uploaded_documents"):
        state["next_action"] = "request_documents"
        return "missing_documents"
    
    return "documentation_processor"

def route_after_documentation(state: AgentState) -> str:
    """Determine next step after documentation processing"""
    
    if state.get("error"):
        return "error"
    
    if state["missing_documents"]:
        state["next_action"] = "missing_documents"
        return "missing_documents"
    
    # Check if all documents are valid
    all_valid = all(doc.validation.is_valid for doc in state["processed_documents"])
    
    if not all_valid:
        state["next_action"] = "invalid_documents"
        return "missing_documents"
    
    return "application_generator"
```

### 9.2 Agent Node Implementations

```python
async def context_builder_node(state: AgentState) -> AgentState:
    """Context builder agent node"""
    try:
        agent = ContextBuilderAgent(llm_service)
        updated_state = await agent.execute(state)
        return updated_state
    except Exception as e:
        state["error"] = str(e)
        return state

async def scheme_researcher_node(state: AgentState) -> AgentState:
    """Scheme researcher agent node"""
    try:
        agent = SchemeResearcherAgent(rag_service, llm_service)
        updated_state = await agent.execute(state)
        return updated_state
    except Exception as e:
        state["error"] = str(e)
        return state

async def eligibility_validator_node(state: AgentState) -> AgentState:
    """Eligibility validator agent node"""
    try:
        agent = EligibilityValidatorAgent(validation_service, llm_service)
        updated_state = await agent.execute(state)
        return updated_state
    except Exception as e:
        state["error"] = str(e)
        return state

async def documentation_processor_node(state: AgentState) -> AgentState:
    """Documentation processor agent node"""
    try:
        agent = DocumentationAgent(ocr_service, validation_service, llm_service)
        updated_state = await agent.execute(state)
        return updated_state
    except Exception as e:
        state["error"] = str(e)
        return state

async def application_generator_node(state: AgentState) -> AgentState:
    """Application generator agent node"""
    try:
        agent = ApplicationReadinessAgent(llm_service)
        updated_state = await agent.execute(state)
        return updated_state
    except Exception as e:
        state["error"] = str(e)
        return state

async def response_formatter_node(state: AgentState) -> AgentState:
    """Format response based on current state"""
    
    next_action = state.get("next_action")
    language = state.get("language", "en")
    
    if next_action == "clarification":
        # Generate clarification questions
        questions = generate_clarification_questions(
            state["missing_attributes"],
            language
        )
        state["response_message"] = questions
        
    elif next_action == "no_schemes_found":
        state["response_message"] = translate(
            "I couldn't find any schemes matching your profile. Let me ask a few more questions...",
            language
        )
        
    elif next_action == "not_eligible":
        result = state["eligibility_result"]
        state["response_message"] = format_ineligibility_message(result, language)
        
    elif next_action == "need_more_info":
        result = state["eligibility_result"]
        state["response_message"] = format_missing_info_message(result, language)
        
    elif next_action == "request_documents":
        scheme = state["current_scheme"]
        state["response_message"] = format_document_request(scheme, language)
        
    elif next_action == "missing_documents":
        state["response_message"] = format_missing_documents_message(
            state["missing_documents"],
            language
        )
        
    else:
        # Success - application ready
        state["response_message"] = format_success_message(
            state["application_json"],
            state["submission_guide"],
            language
        )
    
    return state
```

### 9.3 Parallel Agent Execution

```python
from langgraph.graph import StateGraph
import asyncio

async def parallel_eligibility_check(
    state: AgentState,
    schemes: list[Scheme]
) -> list[EligibilityResult]:
    """Check eligibility for multiple schemes in parallel"""
    
    agent = EligibilityValidatorAgent(validation_service, llm_service)
    
    tasks = []
    for scheme in schemes:
        # Create a copy of state for each scheme
        scheme_state = state.copy()
        scheme_state["current_scheme"] = scheme
        tasks.append(agent.execute(scheme_state))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    
    return [r["eligibility_result"] for r in results]
```

### 9.4 Error Handling and Recovery

```python
class AgentExecutionError(Exception):
    """Base exception for agent execution errors"""
    pass

class LLMServiceError(AgentExecutionError):
    """LLM service failure"""
    pass

class ValidationError(AgentExecutionError):
    """Validation failure"""
    pass

async def execute_with_retry(
    agent_func,
    state: AgentState,
    max_retries: int = 3
) -> AgentState:
    """Execute agent with retry logic"""
    
    for attempt in range(max_retries):
        try:
            return await agent_func(state)
        except LLMServiceError as e:
            if attempt == max_retries - 1:
                # Fallback to rule-based logic
                return await fallback_execution(state)
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except ValidationError as e:
            # No retry for validation errors
            state["error"] = str(e)
            return state
        except Exception as e:
            if attempt == max_retries - 1:
                state["error"] = f"Agent execution failed: {str(e)}"
                return state
            await asyncio.sleep(2 ** attempt)

async def fallback_execution(state: AgentState) -> AgentState:
    """Fallback to rule-based logic when LLM fails"""
    
    # Use rule-based validation only
    if state.get("current_scheme"):
        validation_service = ValidationService()
        result = validation_service.validate_all(
            state["user_context"],
            state["current_scheme"]
        )
        state["eligibility_result"] = result
    
    return state
```

### 9.5 Agent Observability

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

async def traced_agent_execution(
    agent_name: str,
    agent_func,
    state: AgentState
) -> AgentState:
    """Execute agent with tracing"""
    
    with tracer.start_as_current_span(f"agent.{agent_name}") as span:
        span.set_attribute("user_id", state["user_id"])
        span.set_attribute("session_id", state["session_id"])
        span.set_attribute("agent_name", agent_name)
        
        try:
            result = await agent_func(state)
            
            # Log metrics
            if "confidence" in result:
                span.set_attribute("confidence", result["confidence"])
            
            span.set_status(Status(StatusCode.OK))
            return result
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

# Usage
async def context_builder_node(state: AgentState) -> AgentState:
    return await traced_agent_execution(
        "context_builder",
        lambda s: ContextBuilderAgent(llm_service).execute(s),
        state
    )
```

### 9.6 Agent State Persistence

```python
class StatePersistenceService:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def save_state(self, session_id: str, state: AgentState):
        """Save agent state to Redis"""
        key = f"agent_state:{session_id}"
        
        # Serialize state
        state_json = json.dumps(state, default=str)
        
        # Save with TTL (30 minutes)
        await self.redis.setex(key, 1800, state_json)
    
    async def load_state(self, session_id: str) -> AgentState:
        """Load agent state from Redis"""
        key = f"agent_state:{session_id}"
        
        state_json = await self.redis.get(key)
        if not state_json:
            return None
        
        return json.loads(state_json)
    
    async def delete_state(self, session_id: str):
        """Delete agent state"""
        key = f"agent_state:{session_id}"
        await self.redis.delete(key)
```


## 10. Deployment Architecture

### 10.1 Cloud Infrastructure (AWS/GCP)

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER                            │
│                    (ALB / Cloud Load Balancer)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                           │
│              (ECS/Cloud Run - Auto-scaling)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Gateway  │  │ Gateway  │  │ Gateway  │                      │
│  │Instance 1│  │Instance 2│  │Instance 3│                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│              (ECS/Cloud Run - Auto-scaling)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │  Agent   │  │  Agent   │  │  Agent   │                      │
│  │Orchestr. │  │Orchestr. │  │Orchestr. │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│   LLM Service    │ │  RAG Service │ │  OCR Service │
│  (Stateless)     │ │  (Stateless) │ │  (Stateless) │
└──────────────────┘ └──────────────┘ └──────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│   PostgreSQL     │ │  Vector DB   │ │    Redis     │
│   (RDS/Cloud     │ │  (Pinecone/  │ │   (Cache/    │
│     SQL)         │ │   Weaviate)  │ │   Session)   │
└──────────────────┘ └──────────────┘ └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OBJECT STORAGE                              │
│                    (S3 / Cloud Storage)                          │
│                  (Documents, Embeddings)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Container Configuration

#### Dockerfile - API Gateway
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src /app/src
COPY ./config /app/config

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Dockerfile - Agent Orchestrator
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src /app/src
COPY ./config /app/config

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["python", "-m", "src.agents.orchestrator"]
```

### 10.3 Kubernetes Deployment (Optional)

```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  labels:
    app: scheme-copilot
    component: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scheme-copilot
      component: api-gateway
  template:
    metadata:
      labels:
        app: scheme-copilot
        component: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: scheme-copilot/api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-credentials
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
spec:
  selector:
    app: scheme-copilot
    component: api-gateway
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 10.4 Environment Configuration

```yaml
# config/production.yaml
app:
  name: "Government Scheme Copilot"
  environment: "production"
  debug: false
  log_level: "INFO"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60

database:
  host: "${DB_HOST}"
  port: 5432
  name: "${DB_NAME}"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
  pool_size: 20
  max_overflow: 10

redis:
  host: "${REDIS_HOST}"
  port: 6379
  password: "${REDIS_PASSWORD}"
  db: 0
  max_connections: 50

vector_db:
  provider: "pinecone"
  api_key: "${PINECONE_API_KEY}"
  environment: "${PINECONE_ENV}"
  index_name: "schemes"

llm:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4-turbo"
  temperature: 0.7
  max_tokens: 2000
  timeout: 30

embedding:
  provider: "openai"
  model: "text-embedding-3-large"
  dimension: 1536

ocr:
  provider: "tesseract"
  languages: ["eng", "hin"]
  confidence_threshold: 0.7

storage:
  provider: "s3"
  bucket: "${S3_BUCKET}"
  region: "${AWS_REGION}"
  access_key: "${AWS_ACCESS_KEY}"
  secret_key: "${AWS_SECRET_KEY}"

security:
  jwt_secret: "${JWT_SECRET}"
  jwt_algorithm: "HS256"
  jwt_expiration: 3600
  encryption_key: "${ENCRYPTION_KEY}"

rate_limiting:
  enabled: true
  requests_per_minute: 60
  burst: 10

monitoring:
  enabled: true
  sentry_dsn: "${SENTRY_DSN}"
  datadog_api_key: "${DATADOG_API_KEY}"
```

### 10.5 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push API Gateway image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: scheme-copilot-api-gateway
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f Dockerfile.gateway .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
      
      - name: Build and push Agent Orchestrator image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: scheme-copilot-agent-orchestrator
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG -f Dockerfile.agent .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster scheme-copilot-cluster \
            --service api-gateway-service \
            --force-new-deployment
          
          aws ecs update-service \
            --cluster scheme-copilot-cluster \
            --service agent-orchestrator-service \
            --force-new-deployment
```

### 10.6 Infrastructure as Code (Terraform)

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "scheme-copilot-vpc"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier           = "scheme-copilot-db"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.medium"
  allocated_storage    = 100
  storage_type         = "gp3"
  
  db_name  = "scheme_copilot"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "scheme-copilot-final-snapshot"
  
  tags = {
    Name = "scheme-copilot-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "scheme-copilot-redis"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Name = "scheme-copilot-redis"
  }
}

# S3 Bucket for documents
resource "aws_s3_bucket" "documents" {
  bucket = "scheme-copilot-documents-${var.environment}"
  
  tags = {
    Name = "scheme-copilot-documents"
  }
}

resource "aws_s3_bucket_encryption" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "scheme-copilot-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "scheme-copilot-cluster"
  }
}

# ECS Task Definition - API Gateway
resource "aws_ecs_task_definition" "api_gateway" {
  family                   = "scheme-copilot-api-gateway"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "api-gateway"
      image = "${aws_ecr_repository.api_gateway.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        }
      ]
      
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_secretsmanager_secret.db_url.arn
        },
        {
          name      = "OPENAI_API_KEY"
          valueFrom = aws_secretsmanager_secret.openai_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api_gateway.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# ECS Service - API Gateway
resource "aws_ecs_service" "api_gateway" {
  name            = "api-gateway-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api_gateway.arn
  desired_count   = 3
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api_gateway.arn
    container_name   = "api-gateway"
    container_port   = 8000
  }
  
  depends_on = [aws_lb_listener.http]
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "scheme-copilot-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  tags = {
    Name = "scheme-copilot-alb"
  }
}

# Auto Scaling
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 10
  min_capacity       = 3
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api_gateway.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_policy_cpu" {
  name               = "cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

### 10.7 Monitoring and Logging

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

agent_execution_duration = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration',
    ['agent_name']
)

llm_token_usage = Counter(
    'llm_tokens_used_total',
    'Total LLM tokens used',
    ['model', 'operation']
)

active_sessions = Gauge(
    'active_sessions',
    'Number of active user sessions'
)

# Logging configuration
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
})
```


## 11. Scalability Design

### 11.1 Horizontal Scaling Strategy

**API Gateway Layer**:
- Stateless design enables unlimited horizontal scaling
- Auto-scaling based on CPU (>70%) and request rate (>1000 req/min)
- Load balancer distributes traffic across instances

**Agent Orchestrator Layer**:
- Each orchestrator instance handles independent sessions
- Session affinity not required (state stored in Redis)
- Scale based on active session count

**Service Layer**:
- LLM Service: Connection pooling, request queuing
- RAG Service: Parallel vector searches
- OCR Service: Async processing with job queue

### 11.2 Vertical Scaling Considerations

**Database**:
- PostgreSQL: Scale to db.r6g.2xlarge (8 vCPU, 64GB RAM) for production
- Read replicas for analytics queries
- Connection pooling (PgBouncer)

**Redis**:
- Scale to cache.r6g.xlarge (4 vCPU, 26GB RAM)
- Redis Cluster for horizontal scaling if needed

**Vector Database**:
- Pinecone: Serverless auto-scales
- Self-hosted (Weaviate): Scale to 16GB+ RAM for 100K+ documents

### 11.3 Caching Strategy

```python
class CacheService:
    """Multi-layer caching strategy"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache = {}  # In-memory cache
        
    async def get_scheme(self, scheme_id: str) -> Scheme:
        """Get scheme with multi-layer caching"""
        
        # L1: In-memory cache (fastest)
        if scheme_id in self.local_cache:
            return self.local_cache[scheme_id]
        
        # L2: Redis cache
        cached = await self.redis.get(f"scheme:{scheme_id}")
        if cached:
            scheme = Scheme.parse_raw(cached)
            self.local_cache[scheme_id] = scheme
            return scheme
        
        # L3: Database
        scheme = await db.get_scheme(scheme_id)
        
        # Cache for future requests
        await self.redis.setex(
            f"scheme:{scheme_id}",
            3600,  # 1 hour TTL
            scheme.json()
        )
        self.local_cache[scheme_id] = scheme
        
        return scheme
    
    async def get_eligibility_result(
        self,
        user_id: str,
        scheme_id: str
    ) -> EligibilityResult:
        """Cache eligibility results"""
        
        cache_key = f"eligibility:{user_id}:{scheme_id}"
        
        cached = await self.redis.get(cache_key)
        if cached:
            return EligibilityResult.parse_raw(cached)
        
        # Compute eligibility
        result = await compute_eligibility(user_id, scheme_id)
        
        # Cache for 24 hours
        await self.redis.setex(
            cache_key,
            86400,
            result.json()
        )
        
        return result
```

### 11.4 Database Optimization

```sql
-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_schemes_category_state 
ON schemes(category, state) 
WHERE state IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_user_contexts_user_session 
ON user_contexts(user_id, session_id);

CREATE INDEX CONCURRENTLY idx_documents_user_type 
ON documents(user_id, document_type);

CREATE INDEX CONCURRENTLY idx_eligibility_user_scheme 
ON eligibility_results(user_id, scheme_id);

-- Partitioning for messages table (high volume)
CREATE TABLE messages_2026_02 PARTITION OF messages
FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

-- Materialized view for analytics
CREATE MATERIALIZED VIEW scheme_statistics AS
SELECT 
    s.scheme_id,
    s.name,
    COUNT(DISTINCT er.user_id) as eligibility_checks,
    COUNT(DISTINCT CASE WHEN er.is_eligible THEN er.user_id END) as eligible_users,
    AVG(er.confidence) as avg_confidence
FROM schemes s
LEFT JOIN eligibility_results er ON s.scheme_id = er.scheme_id
GROUP BY s.scheme_id, s.name;

CREATE UNIQUE INDEX ON scheme_statistics(scheme_id);

-- Refresh materialized view hourly
REFRESH MATERIALIZED VIEW CONCURRENTLY scheme_statistics;
```

### 11.5 Async Processing with Celery

```python
from celery import Celery

celery_app = Celery(
    'scheme_copilot',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task(bind=True, max_retries=3)
def process_document_async(self, document_id: str):
    """Async document processing"""
    try:
        # OCR processing
        document = get_document(document_id)
        ocr_result = ocr_service.extract_text(document.image_bytes)
        
        # Classification
        doc_type = ocr_service.classify_document(ocr_result.text)
        
        # Field extraction
        fields = ocr_service.extract_fields(ocr_result.text, doc_type)
        
        # Save results
        save_processed_document(document_id, ocr_result, doc_type, fields)
        
        return {"status": "success", "document_id": document_id}
        
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

@celery_app.task
def update_scheme_embeddings(scheme_id: str):
    """Async embedding generation"""
    scheme = get_scheme(scheme_id)
    
    # Process document
    chunks = document_processor.process_scheme_document(scheme)
    
    # Generate embeddings
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Upsert to vector DB
    vector_db.upsert_chunks(chunks, embeddings)
    
    return {"status": "success", "scheme_id": scheme_id}
```

### 11.6 Rate Limiting

```python
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/chat/message")
@limiter.limit("30/minute")
async def chat_message(request: Request, message: ChatMessage):
    """Rate-limited chat endpoint"""
    # Process message
    pass

@app.post("/api/v1/documents/upload")
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile):
    """Rate-limited document upload"""
    # Process upload
    pass

# Custom rate limiter with user-based limits
class UserRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def check_limit(
        self,
        user_id: str,
        action: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if user has exceeded rate limit"""
        
        key = f"rate_limit:{user_id}:{action}"
        
        # Increment counter
        count = await self.redis.incr(key)
        
        # Set expiry on first request
        if count == 1:
            await self.redis.expire(key, window)
        
        return count <= limit
```

### 11.7 Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# PostgreSQL connection pool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Redis connection pool
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    max_connections=50,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

## 12. Security and Privacy Design

### 12.1 Authentication and Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET")
        self.algorithm = "HS256"
        
    def create_access_token(self, user_id: str) -> str:
        """Create JWT access token"""
        payload = {
            "sub": user_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> str:
        """Verify JWT token and return user_id"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            return user_id
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Dependency to get current authenticated user"""
    auth_service = AuthService()
    return auth_service.verify_token(credentials.credentials)

# Usage in endpoints
@app.get("/api/v1/users/me")
async def get_user_profile(user_id: str = Depends(get_current_user)):
    """Protected endpoint"""
    return await get_user(user_id)
```

### 12.2 Data Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

class EncryptionService:
    def __init__(self):
        self.key = self._derive_key(os.getenv("ENCRYPTION_KEY"))
        self.cipher = Fernet(self.key)
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'scheme_copilot_salt',  # Use unique salt per deployment
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Usage
encryption_service = EncryptionService()

# Encrypt before storing
encrypted_aadhaar = encryption_service.encrypt(user.aadhaar_number)
await db.save_user_context(user_id, {"aadhaar": encrypted_aadhaar})

# Decrypt when retrieving
context = await db.get_user_context(user_id)
aadhaar = encryption_service.decrypt(context["aadhaar"])
```

### 12.3 PII Masking

```python
import re

class PIIMasker:
    """Mask PII in logs and responses"""
    
    @staticmethod
    def mask_aadhaar(text: str) -> str:
        """Mask Aadhaar numbers"""
        return re.sub(
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'XXXX-XXXX-****',
            text
        )
    
    @staticmethod
    def mask_phone(text: str) -> str:
        """Mask phone numbers"""
        return re.sub(
            r'\b\d{10}\b',
            'XXXXX*****',
            text
        )
    
    @staticmethod
    def mask_pan(text: str) -> str:
        """Mask PAN numbers"""
        return re.sub(
            r'\b[A-Z]{5}\d{4}[A-Z]\b',
            'XXXXX****X',
            text
        )
    
    @staticmethod
    def mask_all(text: str) -> str:
        """Mask all PII"""
        text = PIIMasker.mask_aadhaar(text)
        text = PIIMasker.mask_phone(text)
        text = PIIMasker.mask_pan(text)
        return text

# Custom logging handler
class PIIMaskingHandler(logging.Handler):
    def emit(self, record):
        record.msg = PIIMasker.mask_all(str(record.msg))
        super().emit(record)
```

### 12.4 Input Validation and Sanitization

```python
from pydantic import BaseModel, validator, Field
import bleach

class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., regex=r'^[a-f0-9\-]{36}$')
    message: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(..., regex=r'^[a-z]{2}$')
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize user input"""
        # Remove HTML tags
        v = bleach.clean(v, tags=[], strip=True)
        # Remove excessive whitespace
        v = ' '.join(v.split())
        return v

class DocumentUploadRequest(BaseModel):
    session_id: str
    file_size: int = Field(..., le=10485760)  # Max 10MB
    mime_type: str = Field(..., regex=r'^image/(jpeg|png|pdf)$')
    
    @validator('mime_type')
    def validate_mime_type(cls, v):
        """Validate file type"""
        allowed_types = ['image/jpeg', 'image/png', 'application/pdf']
        if v not in allowed_types:
            raise ValueError('Invalid file type')
        return v
```

### 12.5 CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://scheme-copilot.gov.in",
        "https://app.scheme-copilot.gov.in"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)
```

### 12.6 Security Headers

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

# HTTPS redirect
app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["scheme-copilot.gov.in", "*.scheme-copilot.gov.in"]
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response
```

### 12.7 Audit Logging

```python
class AuditLogger:
    """Audit logging for sensitive operations"""
    
    def __init__(self, db):
        self.db = db
        
    async def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: dict
    ):
        """Log data access"""
        await self.db.execute(
            """
            INSERT INTO audit_logs (user_id, action, resource, details, timestamp)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            user_id, action, resource, json.dumps(details)
        )
    
    async def log_document_access(self, user_id: str, document_id: str):
        """Log document access"""
        await self.log_access(
            user_id,
            "DOCUMENT_ACCESS",
            f"document:{document_id}",
            {"document_id": document_id}
        )
    
    async def log_eligibility_check(
        self,
        user_id: str,
        scheme_id: str,
        result: bool
    ):
        """Log eligibility check"""
        await self.log_access(
            user_id,
            "ELIGIBILITY_CHECK",
            f"scheme:{scheme_id}",
            {"scheme_id": scheme_id, "eligible": result}
        )
```

## 13. Future Extensions

### 13.1 Government Portal Integration

When APIs become available:
- OAuth integration with government portals
- Automated form submission
- Real-time application status tracking
- Digital signature integration

### 13.2 India Stack Integration

- **Aadhaar eKYC**: Instant identity verification
- **DigiLocker**: Fetch documents directly
- **UPI**: Payment integration for application fees
- **eSign**: Digital signature for applications

### 13.3 Advanced AI Features

- **Fine-tuned LLM**: Domain-specific model for government schemes
- **Multimodal AI**: Process images, videos for documentation
- **Predictive Analytics**: Predict scheme approval probability
- **Personalized Recommendations**: Proactive scheme suggestions

### 13.4 Offline Capabilities

- **Progressive Web App**: Offline-first mobile experience
- **SMS Interface**: USSD-based interaction for feature phones
- **Offline Document Processing**: Edge AI for OCR

### 13.5 Blockchain Integration

- **Document Verification**: Blockchain-based certificate verification
- **Audit Trail**: Immutable application history
- **Smart Contracts**: Automated benefit disbursement

---

## Document Control

**Version**: 1.0  
**Date**: February 15, 2026  
**Status**: Draft  
**Owner**: AI Systems Architecture Team  
**Reviewers**: Engineering, Security, DevOps
