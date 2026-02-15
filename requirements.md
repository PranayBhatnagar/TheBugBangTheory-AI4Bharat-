# Requirements Document: AI-Powered Government Scheme Eligibility & Application Readiness Copilot

## 1. Executive Summary

This document outlines the requirements for an AI-powered, voice-first, multilingual assistant designed to bridge the accessibility gap between rural Indian citizens and government welfare schemes. The system leverages modern AI architecture including LLMs, RAG, agent-based orchestration, and document intelligence to provide conversational scheme discovery, explainable eligibility reasoning, documentation validation, and application readiness generation.

The system does not directly submit applications to government portals due to security, legal, and technical constraints. Instead, it focuses on empowering citizens with knowledge, validation, and application-ready outputs that enable human-assisted submission workflows.

**Target Impact:**
- Enable millions of rural citizens to discover relevant schemes
- Reduce dependency on intermediaries
- Increase scheme enrollment rates
- Provide transparent, explainable eligibility reasoning

## 2. Problem Statement

India operates hundreds of central and state government welfare schemes targeting rural citizens, farmers, entrepreneurs, and vulnerable populations. Despite significant government investment, scheme utilization remains suboptimal due to:

### 2.1 Awareness Gap
- Citizens are unaware of schemes they qualify for
- Information is fragmented across multiple government portals
- No unified discovery mechanism exists

### 2.2 Accessibility Barriers
- Complex eligibility criteria written in legal/bureaucratic language
- Language barriers (English/Hindi vs. regional languages)
- Low digital literacy in rural areas
- No conversational interfaces available

### 2.3 Documentation Challenges
- Citizens struggle to understand required documents
- Lack of validation mechanisms before submission
- High rejection rates due to incomplete/incorrect documentation

### 2.4 Technical Constraints
- Government portals lack public APIs for automated submission
- CAPTCHA, OTP, and security mechanisms prevent automation
- Legal and compliance restrictions on automated submissions

### 2.5 Dependency on Intermediaries
- Citizens rely on middlemen who may charge fees
- Lack of transparency in the application process
- Potential for exploitation

## 3. Objectives

### 3.1 Primary Objectives
1. **Conversational Scheme Discovery**: Enable natural language queries to discover relevant schemes
2. **Explainable Eligibility Assessment**: Provide transparent reasoning for eligibility decisions
3. **Documentation Readiness**: Validate and verify document compliance using AI
4. **Application Readiness**: Generate structured, prefilled application data
5. **Multilingual Support**: Support major Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi)
6. **Voice-First Interface**: Enable voice-based interaction for low-literacy users

### 3.2 Secondary Objectives
1. Build modular, agent-based architecture for extensibility
2. Implement confidence scoring and explainability mechanisms
3. Create human-in-the-loop workflows for assisted submission
4. Design scalable backend infrastructure
5. Ensure data privacy and security compliance

## 4. Stakeholders and Users

### 4.1 Primary Users
- **Rural Citizens**: Farmers, laborers, small business owners, women, elderly
- **Low-Literacy Users**: Individuals requiring voice-based interaction
- **Regional Language Speakers**: Non-Hindi, non-English speakers

### 4.2 Secondary Users
- **Community Helpers**: NGO workers, village volunteers, CSC operators
- **Government Officials**: Scheme administrators, monitoring officers

### 4.3 System Administrators
- **AI Engineers**: System maintenance and model updates
- **Data Curators**: Scheme database management
- **Support Staff**: User assistance and issue resolution

## 5. Functional Requirements

### 5.1 User Authentication and Consent (FR-AUTH)

**FR-AUTH-001**: System shall support phone-based authentication
- OTP verification via SMS
- No mandatory Aadhaar requirement for MVP

**FR-AUTH-002**: System shall capture explicit user consent
- Data usage consent
- Document storage consent
- Communication consent

**FR-AUTH-003**: System shall maintain user session management
- Secure session tokens
- Session timeout after 30 minutes of inactivity

### 5.2 Multimodal Input Interface (FR-INPUT)

**FR-INPUT-001**: System shall accept text-based chat input
- Natural language queries in supported languages
- Structured form input for profile building

**FR-INPUT-002**: System shall accept voice input
- Speech-to-text conversion
- Support for regional accents and dialects

**FR-INPUT-003**: System shall accept document uploads
- Image formats: JPEG, PNG, PDF
- Maximum file size: 10MB per document
- Batch upload support (up to 10 documents)

**FR-INPUT-004**: System shall provide multimodal output
- Text responses
- Text-to-speech for voice users
- Visual cards for scheme information

### 5.3 User Context Building (FR-CONTEXT)

**FR-CONTEXT-001**: System shall extract structured user attributes from conversational input
- Demographics: age, gender, location, caste category
- Economic: income, occupation, land ownership
- Family: household size, dependents, marital status
- Special categories: disability, minority status, BPL status

**FR-CONTEXT-002**: System shall assign confidence scores to extracted attributes
- High confidence (>0.8): Explicitly stated by user
- Medium confidence (0.5-0.8): Inferred from context
- Low confidence (<0.5): Assumed or missing

**FR-CONTEXT-003**: System shall request clarification for low-confidence attributes
- Conversational follow-up questions
- Progressive disclosure to avoid overwhelming users

**FR-CONTEXT-004**: System shall persist user profiles
- Encrypted storage
- Update mechanism for changed circumstances

### 5.4 Scheme Discovery and Retrieval (FR-SCHEME)

**FR-SCHEME-001**: System shall maintain a comprehensive scheme knowledge base
- Central government schemes
- State-specific schemes (starting with 5 major states for MVP)
- Scheme metadata: name, description, benefits, eligibility, documents, application process

**FR-SCHEME-002**: System shall implement RAG-based scheme retrieval
- Vector embeddings of scheme descriptions
- Semantic search over scheme corpus
- Hybrid search (keyword + semantic)

**FR-SCHEME-003**: System shall rank schemes by relevance
- Relevance score based on user context match
- Benefit value estimation
- Application complexity score

**FR-SCHEME-004**: System shall provide scheme summaries
- Plain language descriptions
- Key benefits highlighted
- Estimated benefit amount (if applicable)

### 5.5 Eligibility Validation (FR-ELIGIBILITY)

**FR-ELIGIBILITY-001**: System shall implement hybrid eligibility validation
- Rule-based validation for explicit criteria (age, income thresholds)
- LLM-based reasoning for complex criteria

**FR-ELIGIBILITY-002**: System shall compute eligibility scores
- Binary eligible/not eligible determination
- Confidence score (0-1)
- Partial eligibility indication

**FR-ELIGIBILITY-003**: System shall provide explainable eligibility reasoning
- List of satisfied criteria
- List of unsatisfied criteria
- Missing information requirements

**FR-ELIGIBILITY-004**: System shall handle edge cases
- Conflicting criteria
- Ambiguous user data
- Scheme-specific exceptions

### 5.6 Documentation Validation (FR-DOCS)

**FR-DOCS-001**: System shall perform OCR on uploaded documents
- Extract text from images and PDFs
- Support for Hindi and English text
- Handle poor quality scans

**FR-DOCS-002**: System shall classify document types
- Aadhaar card, PAN card, income certificate, caste certificate, etc.
- Confidence score for classification

**FR-DOCS-003**: System shall validate document completeness
- Check for required fields (name, ID number, dates)
- Verify document validity (expiry dates)
- Flag missing or illegible information

**FR-DOCS-004**: System shall match document data with user profile
- Name matching with fuzzy logic
- Date of birth verification
- Address consistency check

**FR-DOCS-005**: System shall provide document readiness report
- List of validated documents
- Missing documents
- Issues requiring correction

### 5.7 Application Readiness Generation (FR-APPREADY)

**FR-APPREADY-001**: System shall generate structured application data
- JSON format with all required fields
- Prefilled values from user profile and documents
- Placeholder markers for missing data

**FR-APPREADY-002**: System shall generate human-readable application summary
- PDF format
- Checklist of required documents
- Step-by-step submission instructions

**FR-APPREADY-003**: System shall provide portal-specific guidance
- URL of application portal
- Login instructions
- Field mapping guide

**FR-APPREADY-004**: System shall generate printable forms (where applicable)
- Prefilled PDF forms
- Annotations for manual completion

### 5.8 Human-in-the-Loop Workflows (FR-HITL)

**FR-HITL-001**: System shall provide operator dashboard
- Queue of users requiring assistance
- User profile and eligibility summary
- Document viewer

**FR-HITL-002**: System shall enable assisted submission mode
- Screen sharing or guided walkthrough
- Real-time chat with operator
- Session recording for quality assurance

**FR-HITL-003**: System shall track submission status
- Pending, in-progress, submitted, approved, rejected
- Notification mechanism for status updates

### 5.9 Monitoring and Feedback (FR-MONITOR)

**FR-MONITOR-001**: System shall track application lifecycle
- Submission date
- Expected processing time
- Status updates (if available)

**FR-MONITOR-002**: System shall collect user feedback
- Satisfaction rating
- Issue reporting
- Scheme outcome tracking

**FR-MONITOR-003**: System shall provide analytics dashboard
- Scheme discovery metrics
- Eligibility success rates
- Documentation validation accuracy
- User engagement metrics

## 6. Non-Functional Requirements

### 6.1 Performance (NFR-PERF)

**NFR-PERF-001**: Response time for chat interactions shall be <3 seconds
**NFR-PERF-002**: Voice-to-text conversion shall complete within 2 seconds
**NFR-PERF-003**: Document OCR processing shall complete within 10 seconds per document
**NFR-PERF-004**: Scheme retrieval shall return results within 2 seconds
**NFR-PERF-005**: System shall support 1000 concurrent users (MVP target)

### 6.2 Scalability (NFR-SCALE)

**NFR-SCALE-001**: System architecture shall support horizontal scaling
**NFR-SCALE-002**: Database shall handle 1 million user profiles
**NFR-SCALE-003**: Vector database shall support 10,000+ scheme documents
**NFR-SCALE-004**: System shall handle 10,000 daily active users

### 6.3 Availability (NFR-AVAIL)

**NFR-AVAIL-001**: System uptime shall be 99% (MVP target)
**NFR-AVAIL-002**: Graceful degradation for AI service failures
**NFR-AVAIL-003**: Offline mode for basic scheme browsing

### 6.4 Usability (NFR-USABILITY)

**NFR-USABILITY-001**: Interface shall be accessible to users with <5th grade literacy
**NFR-USABILITY-002**: Voice interface shall support regional accents
**NFR-USABILITY-003**: System shall provide contextual help and guidance
**NFR-USABILITY-004**: Error messages shall be in user's preferred language

### 6.5 Security (NFR-SEC)

**NFR-SEC-001**: All data transmission shall use TLS 1.3
**NFR-SEC-002**: User documents shall be encrypted at rest (AES-256)
**NFR-SEC-003**: PII shall be masked in logs and analytics
**NFR-SEC-004**: Role-based access control for operator dashboard
**NFR-SEC-005**: Document retention policy: 90 days after submission

### 6.6 Privacy (NFR-PRIVACY)

**NFR-PRIVACY-001**: System shall comply with Indian data protection regulations
**NFR-PRIVACY-002**: Users shall have right to data deletion
**NFR-PRIVACY-003**: No data sharing with third parties without explicit consent
**NFR-PRIVACY-004**: Audit logs for all data access

### 6.7 Localization (NFR-L10N)

**NFR-L10N-001**: Support for 9 Indian languages (MVP)
**NFR-L10N-002**: Currency display in INR
**NFR-L10N-003**: Date format: DD/MM/YYYY
**NFR-L10N-004**: Regional scheme prioritization based on user location

## 7. AI-Specific Requirements

### 7.1 LLM Requirements (AI-LLM)

**AI-LLM-001**: System shall use production-grade LLM APIs
- OpenAI GPT-4, Google Gemini, or equivalent
- Fallback to secondary provider for reliability

**AI-LLM-002**: System shall implement prompt engineering best practices
- Few-shot examples for structured extraction
- Chain-of-thought prompting for reasoning
- Output format constraints (JSON schema)

**AI-LLM-003**: System shall implement LLM response validation
- Schema validation for structured outputs
- Confidence thresholding
- Fallback to rule-based logic for low-confidence responses

**AI-LLM-004**: System shall handle LLM failures gracefully
- Retry logic with exponential backoff
- Fallback to cached responses
- User notification for degraded service

### 7.2 RAG Requirements (AI-RAG)

**AI-RAG-001**: System shall implement vector-based retrieval
- Embedding model: OpenAI text-embedding-3 or equivalent
- Vector database with similarity search
- Top-k retrieval (k=5-10)

**AI-RAG-002**: System shall implement hybrid search
- Keyword-based search for exact matches
- Semantic search for conceptual matches
- Weighted combination of results

**AI-RAG-003**: System shall implement context window management
- Chunk size optimization (512-1024 tokens)
- Overlap strategy for context preservation
- Dynamic context selection based on query

**AI-RAG-004**: System shall maintain scheme knowledge base freshness
- Quarterly updates for scheme changes
- Version control for scheme documents
- Change detection and notification

### 7.3 Agent Orchestration Requirements (AI-AGENT)

**AI-AGENT-001**: System shall implement agent-based architecture
- Specialized agents for distinct tasks
- Agent communication protocol
- State management across agents

**AI-AGENT-002**: System shall implement agent coordination
- Sequential execution for dependent tasks
- Parallel execution for independent tasks
- Error handling and recovery

**AI-AGENT-003**: System shall implement agent observability
- Logging of agent decisions
- Tracing of agent execution flow
- Performance metrics per agent

### 7.4 Explainability Requirements (AI-EXPLAIN)

**AI-EXPLAIN-001**: System shall provide reasoning for eligibility decisions
- Criteria-level explanations
- Confidence scores for each criterion
- Alternative scenarios ("what-if" analysis)

**AI-EXPLAIN-002**: System shall provide transparency for AI decisions
- Indication when AI vs. rule-based logic is used
- Confidence scores for AI-generated outputs
- Human review option for low-confidence decisions

**AI-EXPLAIN-003**: System shall maintain decision audit trail
- Input data snapshot
- Agent execution log
- Output with reasoning

### 7.5 Model Performance Requirements (AI-PERF)

**AI-PERF-001**: User context extraction accuracy shall be >85%
**AI-PERF-002**: Scheme retrieval relevance (top-5) shall be >80%
**AI-PERF-003**: Eligibility validation accuracy shall be >90%
**AI-PERF-004**: Document classification accuracy shall be >85%
**AI-PERF-005**: OCR text extraction accuracy shall be >90%

## 8. System Constraints

### 8.1 Technical Constraints

**CONST-TECH-001**: No direct integration with government portals
- No public APIs available
- CAPTCHA and OTP prevent automation
- Legal restrictions on automated submissions

**CONST-TECH-002**: Limited scheme data availability
- Manual curation required
- Inconsistent data formats across portals
- Frequent scheme updates

**CONST-TECH-003**: LLM API rate limits and costs
- Token limits per request
- Cost optimization required
- Caching strategy needed

### 8.2 Legal and Compliance Constraints

**CONST-LEGAL-001**: No automated submission to government portals
**CONST-LEGAL-002**: Compliance with Indian data protection laws
**CONST-LEGAL-003**: No guarantee of scheme approval
**CONST-LEGAL-004**: Disclaimer required for AI-generated advice

### 8.3 User Constraints

**CONST-USER-001**: Limited smartphone access in rural areas
**CONST-USER-002**: Intermittent internet connectivity
**CONST-USER-003**: Low digital literacy
**CONST-USER-004**: Limited access to document scanning facilities

### 8.4 Resource Constraints (Hackathon MVP)

**CONST-RESOURCE-001**: Development timeline: 48-72 hours
**CONST-RESOURCE-002**: Limited budget for cloud services
**CONST-RESOURCE-003**: Small team size (3-5 developers)
**CONST-RESOURCE-004**: Limited scheme database (50-100 schemes for MVP)

## 9. Assumptions

### 9.1 User Assumptions

**ASSUME-USER-001**: Users have access to a smartphone or computer
**ASSUME-USER-002**: Users can provide basic personal information
**ASSUME-USER-003**: Users have or can obtain required documents
**ASSUME-USER-004**: Users consent to data collection and processing

### 9.2 Technical Assumptions

**ASSUME-TECH-001**: LLM APIs are available and reliable
**ASSUME-TECH-002**: Cloud infrastructure is available
**ASSUME-TECH-003**: OCR services can handle Indian documents
**ASSUME-TECH-004**: Vector databases can scale to required size

### 9.3 Data Assumptions

**ASSUME-DATA-001**: Scheme eligibility criteria can be structured
**ASSUME-DATA-002**: Scheme information is publicly available
**ASSUME-DATA-003**: Document formats are standardized (Aadhaar, PAN, etc.)
**ASSUME-DATA-004**: User-provided information is truthful

### 9.4 Business Assumptions

**ASSUME-BIZ-001**: Human-assisted submission is acceptable to users
**ASSUME-BIZ-002**: NGOs/CSCs will partner for assisted submission
**ASSUME-BIZ-003**: Government will not block or restrict the system
**ASSUME-BIZ-004**: Users will benefit from application readiness even without direct submission

## 10. MVP Scope

### 10.1 In-Scope for MVP

**MVP-IN-001**: Core Features
- Text-based chat interface (web and mobile-responsive)
- Voice input (speech-to-text)
- User profile building through conversation
- Scheme discovery and retrieval (50-100 schemes)
- Eligibility validation (hybrid approach)
- Document upload and OCR
- Basic document validation
- Application readiness JSON generation
- Multilingual support (Hindi + English + 2 regional languages)

**MVP-IN-002**: Agent Architecture
- User Context Builder Agent
- Scheme Researcher Agent
- Eligibility Validator Agent
- Documentation Agent
- Application Readiness Agent

**MVP-IN-003**: Infrastructure
- FastAPI backend
- React frontend
- PostgreSQL database
- Vector database (Chroma or Pinecone)
- Cloud deployment (single region)

**MVP-IN-004**: Geographic Scope
- 2-3 states (e.g., Maharashtra, Karnataka, Uttar Pradesh)
- Central government schemes

### 10.2 Out-of-Scope for MVP

**MVP-OUT-001**: Direct portal submission
**MVP-OUT-002**: Real-time government portal integration
**MVP-OUT-003**: Operator dashboard (human-in-the-loop)
**MVP-OUT-004**: Application status tracking
**MVP-OUT-005**: SMS/WhatsApp notifications
**MVP-OUT-006**: Offline mobile app
**MVP-OUT-007**: Advanced analytics dashboard
**MVP-OUT-008**: Multi-tenancy for NGO partners
**MVP-OUT-009**: Payment processing
**MVP-OUT-010**: Video KYC integration

### 10.3 MVP Success Criteria

**MVP-SUCCESS-001**: System can handle end-to-end flow for 5 sample schemes
**MVP-SUCCESS-002**: Eligibility validation accuracy >85% on test dataset
**MVP-SUCCESS-003**: Document OCR accuracy >80% on sample documents
**MVP-SUCCESS-004**: Response time <5 seconds for chat interactions
**MVP-SUCCESS-005**: Successful demo with 10 test users
**MVP-SUCCESS-006**: Multilingual support functional for 4 languages

## 11. Future Scope

### 11.1 Phase 2 Enhancements

**FUTURE-P2-001**: Operator dashboard and human-in-the-loop workflows
**FUTURE-P2-002**: Application status tracking (manual updates)
**FUTURE-P2-003**: SMS and WhatsApp integration
**FUTURE-P2-004**: Expanded scheme database (500+ schemes)
**FUTURE-P2-005**: Support for all major Indian languages (20+)
**FUTURE-P2-006**: Mobile app (Android)
**FUTURE-P2-007**: Advanced document validation (fraud detection)
**FUTURE-P2-008**: Scheme recommendation engine (proactive suggestions)

### 11.2 Phase 3 Enhancements

**FUTURE-P3-001**: Government portal integration (if APIs become available)
**FUTURE-P3-002**: Automated submission for select schemes
**FUTURE-P3-003**: Real-time application status tracking
**FUTURE-P3-004**: Video KYC integration
**FUTURE-P3-005**: Biometric authentication
**FUTURE-P3-006**: Blockchain-based document verification
**FUTURE-P3-007**: AI-powered fraud detection
**FUTURE-P3-008**: Predictive analytics for scheme success rates

### 11.3 Long-Term Vision

**FUTURE-LT-001**: National-scale deployment (all states and UTs)
**FUTURE-LT-002**: Integration with India Stack (Aadhaar, DigiLocker)
**FUTURE-LT-003**: Partnership with government for official adoption
**FUTURE-LT-004**: Expansion to other government services (licenses, certificates)
**FUTURE-LT-005**: Open-source platform for other countries
**FUTURE-LT-006**: AI model fine-tuning on Indian government domain
**FUTURE-LT-007**: Federated learning for privacy-preserving model updates
**FUTURE-LT-008**: Voice-only interface for feature phones

---

## Document Control

**Version**: 1.0  
**Date**: February 15, 2026  
**Status**: Draft  
**Owner**: AI Systems Architecture Team  
**Reviewers**: Product, Engineering, Legal, Compliance
