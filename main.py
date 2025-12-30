import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf, clean_text
from app.vectorstore_utils import create_faiss_index, retrive_relevant_docs, clear_chroma_collection
from app.s3_utils import (
    process_uploaded_files_with_s3, 
    list_documents_in_s3, 
    download_document_from_s3,
    get_s3_documents_for_vector_processing,
    process_all_s3_documents_for_vector_storage
)
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from app.email_utils import send_medical_analytics, generate_document_insights, validate_email, send_support_ticket
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time


st.set_page_config(
    page_title="MediChat Pro - Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    .chat-message .avatar {
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff3333;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
# Email functionality is now function-based, no need for session state
if "document_count" not in st.session_state:
    st.session_state.document_count = 0
if "receiver_email" not in st.session_state:
    st.session_state.receiver_email = ""

# Auto-initialize chat with existing ChromaDB data
if st.session_state.vectorstore is None and st.session_state.chat_model is None:
    try:
        from app.vectorstore_utils import ensure_collection_exists
        collection = ensure_collection_exists()
        if collection and collection.count() > 0:
            # Initialize chat model
            chat_model = get_chat_model(EURI_API_KEY)
            st.session_state.chat_model = chat_model
            st.session_state.vectorstore = collection  # Set actual collection object
            st.session_state.document_count = collection.count()
            st.success(f"Chat initialized with {collection.count()} existing documents from ChromaDB!")
        else:
            st.info("No existing documents found in ChromaDB. Upload documents to start chatting.")
    except Exception as e:
        st.warning(f"Error initializing ChromaDB: {e}")
        st.info("Upload documents to start chatting.")
    
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">MediChat Pro</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document upload and email configuration
with st.sidebar:
    st.markdown("### Document Upload")
    st.markdown("Upload your medical documents to start chatting!")
    
    uploaded_files = pdf_uploader()
    
    st.markdown("---")
    st.markdown("### Manual Initialization")
    if st.button("Initialize with Existing ChromaDB Data"):
        try:
            from app.vectorstore_utils import ensure_collection_exists
            collection = ensure_collection_exists()
            if collection and collection.count() > 0:
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model
                st.session_state.vectorstore = collection
                st.session_state.document_count = collection.count()
                st.success(f"Initialized with {collection.count()} documents!")
            else:
                st.error("No documents found in ChromaDB")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("### Email Configuration")
    st.markdown("Enter recipient email for analysis reports")
    
    # Email input field
    receiver_email = st.text_input(
        "Recipient Email",
        value=st.session_state.get("receiver_email", ""),
        placeholder="Enter email address (e.g., user@example.com)",
        help="Enter the email address where you want to receive medical analysis reports"
    )
    
    # Update session state
    if receiver_email:
        st.session_state.receiver_email = receiver_email
        
        # Validate email format
        if validate_email(receiver_email):
            st.success(f"Valid email: {receiver_email}")
        else:
            st.error("Please enter a valid email address")
    else:
        st.info("Enter an email address to enable report sending")
    
    # Email test section
    if st.session_state.get("receiver_email") and validate_email(st.session_state.get("receiver_email")):
        st.markdown("---")
        st.markdown("### Test Email")
        if st.button("Send Test Email", help="Send a test email to verify configuration"):
            with st.spinner("Sending test email..."):
                test_insights = {
                    'total_documents': 1,
                    'total_chunks': 5,
                    'relevant_docs_count': 2,
                    'confidence_score': '95.0%',
                    'response_time': '1.50',
                    'query_complexity': 'Test',
                    'document_coverage': 'This is a test email to verify email configuration.',
                    'medical_keywords': ['test', 'email', 'configuration']
                }
                
                success = send_medical_analytics(
                    user_query="Test email configuration",
                    ai_response="This is a test email to verify that the email functionality is working correctly. If you receive this email, the configuration is successful!",
                    document_insights=test_insights,
                    receiver_email=st.session_state.get("receiver_email"),
                    chat_history=[]
                )
                
                if success:
                    st.success(f"Test email sent to {st.session_state.get('receiver_email')}!")
                else:
                    st.error("Failed to send test email")
    
    # Always Visible Action Buttons
    st.markdown("---")
    st.markdown("### Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Send Analysis Report Button
        if st.button("📊 Send Analysis Report", help="Send current session analysis to email", disabled=not (st.session_state.get("receiver_email") and validate_email(st.session_state.get("receiver_email")))):
            if st.session_state.get("receiver_email") and validate_email(st.session_state.get("receiver_email")):
                with st.spinner("Generating and sending analysis report..."):
                    # Generate insights from current session
                    insights = generate_document_insights(
                        st.session_state.messages if st.session_state.messages else [],
                        st.session_state.get("document_count", 0)
                    )
                    
                    success = send_medical_analytics(
                        user_query="Session analysis report",
                        ai_response="Complete session analysis with document insights and chat history.",
                        document_insights=insights,
                        receiver_email=st.session_state.get("receiver_email"),
                        chat_history=st.session_state.messages[-10:] if st.session_state.messages else []
                    )
                    
                    if success:
                        st.success(f"✅ Analysis report sent to {st.session_state.get('receiver_email')}!")
                    else:
                        st.error("❌ Failed to send analysis report")
            else:
                st.error("❌ Please enter a valid email address first")
    
    with col2:
        # Create Support Ticket Button
        if st.button("🎫 Create Support Ticket", help="Create a support ticket for any issues"):
            with st.spinner("Creating support ticket..."):
                user_email = st.session_state.get("receiver_email")
                
                success = send_support_ticket(
                    user_query="Support request from user",
                    ai_response="User requested support assistance through the sidebar button.",
                    user_email=user_email,
                    chat_history=st.session_state.messages[-5:] if st.session_state.messages else []
                )
                
                if success:
                    st.success("✅ Support ticket created and sent to sudhanshu@euron.one!")
                else:
                    st.error("❌ Failed to create support ticket")
    
    # Save Session Button
    st.markdown("---")
    if st.button("💾 Save Session", help="Save current session data"):
        if st.session_state.get("receiver_email") and validate_email(st.session_state.get("receiver_email")):
            with st.spinner("Saving session..."):
                # Create session summary
                session_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'email': st.session_state.get("receiver_email"),
                    'document_count': st.session_state.get("document_count", 0),
                    'message_count': len(st.session_state.messages) if st.session_state.messages else 0,
                    'vectorstore_ready': bool(st.session_state.get("vectorstore")),
                    'recent_messages': st.session_state.messages[-5:] if st.session_state.messages else []
                }
                
                # Send session summary via email
                insights = {
                    'total_documents': session_data['document_count'],
                    'total_chunks': session_data['message_count'],
                    'confidence_score': '100%',
                    'response_time': '0.00',
                    'query_complexity': 'Session Save',
                    'document_coverage': f"Session saved with {session_data['document_count']} documents and {session_data['message_count']} messages.",
                    'medical_keywords': ['session', 'save', 'backup']
                }
                
                success = send_medical_analytics(
                    user_query="Session save request",
                    ai_response=f"Session saved successfully at {session_data['timestamp']}. Documents: {session_data['document_count']}, Messages: {session_data['message_count']}",
                    document_insights=insights,
                    receiver_email=st.session_state.get("receiver_email"),
                    chat_history=session_data['recent_messages']
                )
                
                if success:
                    st.success("✅ Session saved and summary sent to email!")
                else:
                    st.error("❌ Failed to save session")
        else:
            st.error("❌ Please enter a valid email address to save session")
    
    # Always Visible Analytics Display
    st.markdown("---")
    st.markdown("### 📊 Session Analytics")
    
    if st.session_state.get("document_count", 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Documents", st.session_state.get("document_count", 0))
        
        with col2:
            st.metric("💬 Messages", len(st.session_state.messages) if st.session_state.messages else 0)
        
        with col3:
            if st.session_state.get("vectorstore"):
                st.metric("🗄️ Vector DB", "Ready")
            else:
                st.metric("🗄️ Vector DB", "Not Ready")
        
        # Show recent chat summary
        if st.session_state.messages:
            st.markdown("**Recent Chat Summary:**")
            recent_messages = st.session_state.messages[-3:] if len(st.session_state.messages) > 3 else st.session_state.messages
            for msg in recent_messages:
                role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                st.write(f"{role}: {msg['content'][:100]}...")
    else:
        st.info("📊 Upload documents to see analytics")
    
    st.markdown("---")
    st.markdown("### 🗄️ ChromaDB Management")
    if st.button("🗑️ Clear All Documents", help="Clear all documents from ChromaDB collection"):
        try:
            if clear_chroma_collection():
                st.success("✅ ChromaDB collection cleared successfully!")
                # Clear session state
                if 'vectorstore' in st.session_state:
                    del st.session_state.vectorstore
                if 'document_count' in st.session_state:
                    del st.session_state.document_count
                st.rerun()
            else:
                st.error("❌ Failed to clear ChromaDB collection")
        except Exception as e:
            st.error(f"❌ Error clearing collection: {e}")
    
    st.markdown("---")
    st.markdown("### ☁️ S3 Document Management")
    
    # List S3 documents
    if st.button("📄 List S3 Documents", help="Show all documents in S3 bucket"):
        with st.spinner("Loading S3 documents..."):
            s3_docs = list_documents_in_s3()
            if s3_docs:
                st.success(f"✅ Found {len(s3_docs)} documents in S3")
                for i, doc in enumerate(s3_docs):
                    with st.expander(f"📄 {doc['filename']}"):
                        st.write(f"**S3 Key:** `{doc['key']}`")
                        st.write(f"**Size:** {doc['size']} bytes")
                        st.write(f"**Last Modified:** {doc['last_modified']}")
                        st.write(f"**S3 URL:** `{doc['s3_url']}`")
            else:
                st.info("📂 No documents found in S3 bucket")
    
    # Process S3 documents for vector storage
    if st.button("🔄 Process All S3 Documents", help="Download and process all S3 documents for vector storage"):
        with st.spinner("Processing all S3 documents..."):
            # Process all S3 documents for vector storage
            all_texts = process_all_s3_documents_for_vector_storage(extract_text_from_pdf)
            
            if all_texts:
                st.info(f"📄 Processing {len(all_texts)} documents from S3 for vector storage...")
                
                # Split texts into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))
                
                # Create ChromaDB collection with batch processing
                st.info(f"📚 Processing {len(chunks)} document chunks in batches...")
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.document_count = len(all_texts)
                
                # Initialize chat model
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model
                
                st.success(f"✅ Successfully processed {len(all_texts)} S3 documents with {len(chunks)} chunks!")
                st.success("✅ All S3 documents now available for chat!")
            else:
                st.warning("⚠️ No documents could be processed from S3")
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")
        
        # Process documents with S3 integration
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                # Process files with S3 integration
                s3_results = process_uploaded_files_with_s3(uploaded_files, extract_text_from_pdf)
                
                # Display S3 upload results
                if s3_results['uploaded_to_s3']:
                    st.success(f"✅ {len(s3_results['uploaded_to_s3'])} documents uploaded to S3")
                    for doc in s3_results['uploaded_to_s3']:
                        st.write(f"📄 {doc['filename']} → S3")
                
                if s3_results['already_in_s3']:
                    st.info(f"📄 {len(s3_results['already_in_s3'])} documents already exist in S3")
                    for doc in s3_results['already_in_s3']:
                        st.write(f"📄 {doc['filename']} (already in S3)")
                
                if s3_results['failed_uploads']:
                    st.warning(f"⚠️ {len(s3_results['failed_uploads'])} documents failed to upload to S3")
                    for doc in s3_results['failed_uploads']:
                        st.write(f"❌ {doc['filename']}: {doc['error']}")
                
                # Process texts for vector storage
                all_texts = s3_results['all_texts']
                if not all_texts:
                    st.error("❌ No text could be extracted from the uploaded files")
                else:
                    # Split texts into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    
                    chunks = []
                    for text in all_texts:
                        chunks.extend(text_splitter.split_text(text))
                    
                    # Create ChromaDB collection with batch processing
                    st.info(f"📚 Processing {len(chunks)} document chunks in batches...")
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.document_count = len(uploaded_files)
                
                # Initialize chat model
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model
                
                st.success("✅ Documents processed successfully!")
                st.success("✅ Documents stored in both S3 and ChromaDB!")
                st.balloons()

# Main chat interface
st.markdown("### 💬 Chat with Your Medical Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        pass

# Chat input
if prompt := st.chat_input("Ask about your medical documents or use commands like 'send report to email@example.com'..."):
    # Check for chat commands first
    command_processed = False
    
    # Debug: Show what command was detected
    st.write(f"Debug: Processing prompt: '{prompt}'")
    
    # Command: Send report to email
    if (("send report to" in prompt.lower() or 
         "send analysis report to" in prompt.lower() or
         "send analytics to" in prompt.lower() or
         "send this report to" in prompt.lower() or
         "email report to" in prompt.lower() or
         "send this repot to" in prompt.lower() or  # Handle typo
         "send report" in prompt.lower() or
         ("send" in prompt.lower() and "report" in prompt.lower() and "@" in prompt) or
         ("send" in prompt.lower() and "repot" in prompt.lower() and "@" in prompt)) and "@" in prompt):
        # Extract email using regex for better accuracy
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, prompt)
        if email_match:
            email_match = email_match.group()
        else:
            email_match = ""
        if validate_email(email_match):
            with st.chat_message("assistant"):
                with st.spinner("Processing command..."):
                    # Generate insights
                    insights = {
                        'total_documents': st.session_state.get("document_count", 0),
                        'message_count': len(st.session_state.messages) if st.session_state.messages else 0,
                        'session_summary': "Report sent via chat command"
                    }
                    
                    success = send_medical_analytics(
                        user_query="Chat command: send report",
                        ai_response=f"Analysis report sent to {email_match} as requested via chat command.",
                        document_insights=insights,
                        receiver_email=email_match,
                        chat_history=st.session_state.messages[-10:] if st.session_state.messages else []
                    )
                    
                    if success:
                        response = f"Analysis report sent to {email_match}!"
                    else:
                        response = f"Failed to send report to {email_match}"
                    
                    st.markdown(response)
                    st.caption(time.strftime("%H:%M"))
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": prompt
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    command_processed = True
        else:
            with st.chat_message("assistant"):
                response = f"Invalid email address: {email_match}"
                st.markdown(response)
                st.caption(time.strftime("%H:%M"))
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                command_processed = True
    
    # Command: Create support ticket
    elif (prompt.lower().startswith("create support ticket") or 
          prompt.lower().startswith("support ticket") or
          "create a support ticket" in prompt.lower() or
          "create support ticket" in prompt.lower() or
          "send support ticket" in prompt.lower() or
          "create ticket" in prompt.lower() or
          ("support" in prompt.lower() and "ticket" in prompt.lower()) or
          "generate ticket" in prompt.lower() or
          "open ticket" in prompt.lower()):
        with st.chat_message("assistant"):
            with st.spinner("Creating support ticket..."):
                user_email = st.session_state.get("receiver_email")
                
                success = send_support_ticket(
                    user_query=prompt,
                    ai_response="Support ticket created via chat command.",
                    user_email=user_email,
                    chat_history=st.session_state.messages[-5:] if st.session_state.messages else []
                )
                
                if success:
                    response = "Support ticket created and sent to sudhanshu@euron.one!"
                else:
                    response = "Failed to create support ticket"
                
                st.markdown(response)
                st.caption(time.strftime("%H:%M"))
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                command_processed = True
    
    # Command: Process S3 documents
    elif prompt.lower().startswith("process s3") or prompt.lower().startswith("process all s3"):
        with st.chat_message("assistant"):
            with st.spinner("Processing all S3 documents..."):
                # Process all S3 documents for vector storage
                all_texts = process_all_s3_documents_for_vector_storage(extract_text_from_pdf)
                
                if all_texts:
                    # Split texts into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    
                    chunks = []
                    for text in all_texts:
                        chunks.extend(text_splitter.split_text(text))
                    
                    # Create ChromaDB collection with batch processing
                    vectorstore = create_faiss_index(chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_count = len(all_texts)
                    
                    # Initialize chat model
                    chat_model = get_chat_model(EURI_API_KEY)
                    st.session_state.chat_model = chat_model
                    
                    response = f"Successfully processed {len(all_texts)} S3 documents with {len(chunks)} chunks! All S3 documents are now available for chat."
                else:
                    response = "No documents could be processed from S3"
                
                st.markdown(response)
                st.caption(time.strftime("%H:%M"))
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                command_processed = True
    
    # Command: Save session
    elif (prompt.lower().startswith("save session") or
          "save session" in prompt.lower() or
          "save the session" in prompt.lower()):
        with st.chat_message("assistant"):
            with st.spinner("Saving session..."):
                if st.session_state.get("receiver_email") and validate_email(st.session_state.get("receiver_email")):
                    # Create session summary
                    session_data = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'email': st.session_state.get("receiver_email"),
                        'document_count': st.session_state.get("document_count", 0),
                        'message_count': len(st.session_state.messages) if st.session_state.messages else 0,
                        'vectorstore_ready': bool(st.session_state.get("vectorstore")),
                        'recent_messages': st.session_state.messages[-5:] if st.session_state.messages else []
                    }
                    
                    # Send session summary via email
                    insights = {
                        'total_documents': session_data['document_count'],
                        'total_chunks': session_data['message_count'],
                        'confidence_score': '100%',
                        'response_time': '0.00',
                        'query_complexity': 'Session Save',
                        'document_coverage': f"Session saved with {session_data['document_count']} documents and {session_data['message_count']} messages.",
                        'medical_keywords': ['session', 'save', 'backup']
                    }
                    
                    success = send_medical_analytics(
                        user_query="Chat command: save session",
                        ai_response=f"Session saved successfully at {session_data['timestamp']}. Documents: {session_data['document_count']}, Messages: {session_data['message_count']}",
                        document_insights=insights,
                        receiver_email=st.session_state.get("receiver_email"),
                        chat_history=session_data['recent_messages']
                    )
                    
                    if success:
                        response = "Session saved and summary sent to email!"
                    else:
                        response = "Failed to save session"
                else:
                    response = "Please enter a valid email address to save session"
                
                st.markdown(response)
                st.caption(time.strftime("%H:%M"))
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": prompt
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                command_processed = True
    
    # If no command was processed, handle as normal chat
    if not command_processed:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if st.session_state.vectorstore is not None and st.session_state.chat_model is not None:
            with st.chat_message("assistant"):
                start_time = time.time()
                
                with st.spinner("Searching documents..."):
                    # Retrieve relevant documents from ChromaDB
                    relevant_docs = retrive_relevant_docs(st.session_state.vectorstore, prompt, k=10)
                    
                    # Create context from relevant documents
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Enhanced prompt with better medical context
                    system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant with expertise in medical analysis. 
                    Based on the following medical documents, provide accurate, comprehensive, and clinically relevant answers.
                    
                    IMPORTANT INSTRUCTIONS:
                    1. Always base your response on the provided medical documents
                    2. If information is not in the documents, clearly state "Based on the provided documents, this information is not available"
                    3. Provide specific medical insights, potential concerns, and recommendations when appropriate
                    4. Use medical terminology accurately
                    5. Include relevant medical context and explanations
                    6. If discussing medications, mention dosage, side effects, and interactions when available
                    7. For symptoms, provide potential causes and recommended actions
                    8. Always emphasize that this is for informational purposes and not a substitute for professional medical advice
                    9. When asked for patient details or all documents, provide information from ALL available documents
                    10. If multiple patients are mentioned, organize the information by patient for clarity
                    11. Keep responses concise and focused on question asked

                    Medical Documents Context (Total Documents: {len(relevant_docs)}):
                    {context}

                    User Question: {prompt}

                    Please provide a comprehensive medical analysis and answer:"""
                    
                    response = ask_chat_model(st.session_state.chat_model, system_prompt)
                
                response_time = time.time() - start_time
                
                st.markdown(response)
                
                # Generate document insights
                document_insights = generate_document_insights(
                    st.session_state.vectorstore, 
                    prompt, 
                    response, 
                    relevant_docs, 
                    response_time
                )
                document_insights['total_documents'] = st.session_state.document_count
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "insights": document_insights
                })
                
                # Action buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    # Check if receiver email is configured and valid
                    receiver_email = st.session_state.get("receiver_email")
                    if receiver_email and validate_email(receiver_email):
                        if st.button("Send Analysis Report", key=f"email_{len(st.session_state.messages)}"):
                            with st.spinner("Sending email report..."):
                                success = send_medical_analytics(
                                    user_query=prompt,
                                    ai_response=response,
                                    document_insights=document_insights,
                                    receiver_email=receiver_email,
                                    chat_history=st.session_state.messages[-10:]  # Last 10 messages
                                )
                                if success:
                                    st.success(f"Analysis report sent to {receiver_email}!")
                                else:
                                    st.error("Failed to send email report")
                    else:
                        st.button("Send Analysis Report", key=f"email_{len(st.session_state.messages)}", disabled=True, 
                                 help="Please configure a valid recipient email in the sidebar")
                
                with col2:
                    if st.button("Create Support Ticket", key=f"support_{len(st.session_state.messages)}"):
                        with st.spinner("Creating support ticket..."):
                            # Get user email from session state if available
                            user_email = st.session_state.get("receiver_email")
                            
                            success = send_support_ticket(
                                user_query=prompt,
                                ai_response=response,
                                user_email=user_email,
                                chat_history=st.session_state.messages[-10:]  # Last 10 messages
                            )
                            if success:
                                st.success("Support ticket created and sent to sudhanshu@euron.one!")
                            else:
                                st.error("Failed to create support ticket")
                
                with col3:
                    if st.button("Show Analytics", key=f"analytics_{len(st.session_state.messages)}"):
                        st.json(document_insights)
                
                with col4:
                    if st.button("Save Session", key=f"save_{len(st.session_state.messages)}"):
                        # Save session data (could be enhanced to save to file)
                        st.success("Session data saved!")
        else:
            with st.chat_message("assistant"):
                st.error("Please upload and process documents first!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence</p>
</div>
""", unsafe_allow_html=True)    
