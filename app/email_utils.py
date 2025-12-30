import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import streamlit as st
import re
from app.config import EMAIL_CONFIG


def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def create_medical_analytics_email(user_query: str, 
                                 ai_response: str, 
                                 document_insights: Dict[str, Any],
                                 receiver_email: str,
                                 chat_history: List[Dict] = None) -> MIMEMultipart:
    """Create a comprehensive medical analytics email"""
    
    config = EMAIL_CONFIG
    msg = MIMEMultipart('alternative')
    msg['From'] = config['sender_email']
    msg['To'] = receiver_email
    msg['Subject'] = f"MediChat Pro - Medical Document Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create HTML content
    html_content = generate_html_email_content(
        user_query, ai_response, document_insights, chat_history
    )
    
    # Create plain text content
    text_content = generate_text_email_content(
        user_query, ai_response, document_insights, chat_history
    )
    
    # Attach both versions
    part1 = MIMEText(text_content, 'plain')
    part2 = MIMEText(html_content, 'html')
    
    msg.attach(part1)
    msg.attach(part2)
    
    return msg


def generate_html_email_content(user_query: str, 
                              ai_response: str, 
                              document_insights: Dict[str, Any],
                              chat_history: List[Dict] = None) -> str:
    """Generate HTML email content"""
    
    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #ff4b4b; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #ff4b4b; border-bottom: 2px solid #ff4b4b; padding-bottom: 10px; }}
                .query-box {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #ff4b4b; margin: 15px 0; }}
                .response-box {{ background-color: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
                .insights-box {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
                .chat-history {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .chat-message {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
                .user-message {{ background-color: #e3f2fd; }}
                .assistant-message {{ background-color: #f3e5f5; }}
                .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; color: #666; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏥 MediChat Pro - Medical Document Analysis</h1>
                <p>Intelligent Medical Document Assistant Report</p>
            </div>
            
            <div class="content">
                <div class="section">
                    <h2>📋 Session Summary</h2>
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Documents Processed:</strong> {document_insights.get('total_documents', 'N/A')}</p>
                    <p><strong>Total Chunks:</strong> {document_insights.get('total_chunks', 'N/A')}</p>
                </div>
                
                <div class="section">
                    <h2>❓ User Query</h2>
                    <div class="query-box">
                        <strong>Question:</strong> {user_query}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🤖 AI Response</h2>
                    <div class="response-box">
                        {ai_response.replace(chr(10), '<br>')}
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Document Analytics</h2>
                    <div class="insights-box">
                        <h3>Key Insights:</h3>
                        <ul>
                            <li><strong>Relevant Documents Found:</strong> {document_insights.get('relevant_docs_count', 'N/A')}</li>
                            <li><strong>Confidence Score:</strong> {document_insights.get('confidence_score', 'N/A')}</li>
                            <li><strong>Response Time:</strong> {document_insights.get('response_time', 'N/A')} seconds</li>
                            <li><strong>Query Complexity:</strong> {document_insights.get('query_complexity', 'N/A')}</li>
                        </ul>
                        
                        <h3>Document Coverage:</h3>
                        <p>{document_insights.get('document_coverage', 'Analysis of document coverage not available.')}</p>
                        
                        <h3>Medical Keywords Detected:</h3>
                        <p>{', '.join(document_insights.get('medical_keywords', ['None detected']))}</p>
                    </div>
                </div>
        """
        
    # Add chat history if available
    if chat_history and len(chat_history) > 0:
        html += """
                <div class="section">
                    <h2>💬 Chat History</h2>
                    <div class="chat-history">
            """
        for message in chat_history[-5:]:  # Show last 5 messages
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            
            if role == 'user':
                html += f'<div class="chat-message user-message"><strong>User ({timestamp}):</strong> {content}</div>'
            else:
                html += f'<div class="chat-message assistant-message"><strong>Assistant ({timestamp}):</strong> {content}</div>'
        
        html += """
                </div>
            </div>
        """
        
        html += """
                <div class="section">
                    <h2>🔍 Technical Details</h2>
                    <div class="metric">Model: GPT-4.1-nano</div>
                    <div class="metric">Embedding: sentence-transformers/all-mpnet-base-v2</div>
                    <div class="metric">Vector Store: FAISS</div>
                    <div class="metric">Framework: LangChain</div>
                </div>
            </div>
            
            <div class="footer">
                <p>🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence</p>
                <p>This report was automatically generated by MediChat Pro</p>
            </div>
        </body>
        </html>
        """
        
    return html


def generate_text_email_content(user_query: str, 
                              ai_response: str, 
                              document_insights: Dict[str, Any],
                              chat_history: List[Dict] = None) -> str:
    """Generate plain text email content"""
    
    text = f"""
MEDICHAT PRO - MEDICAL DOCUMENT ANALYSIS REPORT
==============================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Documents Processed: {document_insights.get('total_documents', 'N/A')}
Total Chunks: {document_insights.get('total_chunks', 'N/A')}

USER QUERY:
-----------
{user_query}

AI RESPONSE:
------------
{ai_response}

DOCUMENT ANALYTICS:
------------------
Relevant Documents Found: {document_insights.get('relevant_docs_count', 'N/A')}
Confidence Score: {document_insights.get('confidence_score', 'N/A')}
Response Time: {document_insights.get('response_time', 'N/A')} seconds
Query Complexity: {document_insights.get('query_complexity', 'N/A')}

Document Coverage:
{document_insights.get('document_coverage', 'Analysis of document coverage not available.')}

Medical Keywords Detected:
{', '.join(document_insights.get('medical_keywords', ['None detected']))}
        """
        
    if chat_history and len(chat_history) > 0:
        text += "\n\nCHAT HISTORY:\n------------\n"
        for message in chat_history[-5:]:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            text += f"{role.upper()} ({timestamp}): {content}\n"
        
        text += """
TECHNICAL DETAILS:
-----------------
Model: GPT-4.1-nano
Embedding: sentence-transformers/all-mpnet-base-v2
Vector Store: FAISS
Framework: LangChain

---
🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence
This report was automatically generated by MediChat Pro
        """
        
    return text


def send_email(message: MIMEMultipart) -> bool:
    """Send email using SMTP"""
    try:
        config = EMAIL_CONFIG
        # Create secure connection
        context = ssl.create_default_context()
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls(context=context)
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(message)
        
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False


def send_medical_analytics(user_query: str, 
                         ai_response: str, 
                         document_insights: Dict[str, Any],
                         receiver_email: str,
                         chat_history: List[Dict] = None) -> bool:
    """Send medical analytics email"""
    try:
        message = create_medical_analytics_email(
            user_query, ai_response, document_insights, receiver_email, chat_history
        )
        return send_email(message)
    except Exception as e:
        st.error(f"Failed to create or send medical analytics email: {str(e)}")
        return False


def create_support_ticket_email(user_query: str, 
                              ai_response: str, 
                              user_email: str = None,
                              chat_history: List[Dict] = None) -> MIMEMultipart:
    """Create a support ticket email"""
    
    config = EMAIL_CONFIG
    msg = MIMEMultipart('alternative')
    msg['From'] = config['sender_email']
    msg['To'] = 'sourabh.work2308@gmail.com'  # Support email
    msg['Subject'] = f"MediChat Pro Support Ticket - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create HTML content
    html_content = generate_support_ticket_html_content(
        user_query, ai_response, user_email, chat_history
    )
    
    # Create plain text content
    text_content = generate_support_ticket_text_content(
        user_query, ai_response, user_email, chat_history
    )
    
    # Attach both versions
    part1 = MIMEText(text_content, 'plain')
    part2 = MIMEText(html_content, 'html')
    
    msg.attach(part1)
    msg.attach(part2)
    
    return msg


def generate_support_ticket_html_content(user_query: str, 
                                       ai_response: str, 
                                       user_email: str = None,
                                       chat_history: List[Dict] = None) -> str:
    """Generate HTML content for support ticket"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #ff4b4b; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #ff4b4b; border-bottom: 2px solid #ff4b4b; padding-bottom: 10px; }}
            .query-box {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
            .response-box {{ background-color: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
            .support-info {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #17a2b8; margin: 15px 0; }}
            .chat-history {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .chat-message {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
            .user-message {{ background-color: #e3f2fd; }}
            .assistant-message {{ background-color: #f3e5f5; }}
            .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; color: #666; }}
            .priority {{ display: inline-block; margin: 10px; padding: 10px; background-color: #ffc107; border-radius: 5px; color: #000; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎫 MediChat Pro Support Ticket</h1>
            <p>User Support Request</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📋 Ticket Information</h2>
                <p><strong>Ticket ID:</strong> MC-{datetime.now().strftime('%Y%m%d%H%M%S')}</p>
                <p><strong>Created:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Priority:</strong> <span class="priority">MEDIUM</span></p>
                <p><strong>Category:</strong> User Query Support</p>
            </div>
            
            <div class="section">
                <h2>👤 User Information</h2>
                <div class="support-info">
                    <p><strong>User Email:</strong> {user_email if user_email else 'Not provided'}</p>
                    <p><strong>Platform:</strong> MediChat Pro Web Application</p>
                    <p><strong>Session Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>❓ User Query</h2>
                <div class="query-box">
                    <strong>Question:</strong> {user_query}
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 AI Response</h2>
                <div class="response-box">
                    {ai_response.replace(chr(10), '<br>')}
                </div>
            </div>
    """
    
    # Add chat history if available
    if chat_history and len(chat_history) > 0:
        html += """
            <div class="section">
                <h2>💬 Chat History Context</h2>
                <div class="chat-history">
        """
        for message in chat_history[-10:]:  # Show last 10 messages
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            
            if role == 'user':
                html += f'<div class="chat-message user-message"><strong>User ({timestamp}):</strong> {content}</div>'
            else:
                html += f'<div class="chat-message assistant-message"><strong>Assistant ({timestamp}):</strong> {content}</div>'
        
        html += """
                </div>
            </div>
        """
    
    html += """
            <div class="section">
                <h2>🔍 Support Notes</h2>
                <div class="support-info">
                    <p><strong>Action Required:</strong> Please review the user query and AI response above.</p>
                    <p><strong>Context:</strong> This ticket was automatically generated from a user's chat interaction with MediChat Pro.</p>
                    <p><strong>Next Steps:</strong> Contact the user if additional information is needed or if the AI response requires human review.</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>🎫 MediChat Pro Support System | 🤖 Automated Ticket Generation</p>
            <p>This support ticket was automatically generated by the MediChat Pro application</p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_support_ticket_text_content(user_query: str, 
                                       ai_response: str, 
                                       user_email: str = None,
                                       chat_history: List[Dict] = None) -> str:
    """Generate plain text content for support ticket"""
    
    text = f"""
MEDICHAT PRO SUPPORT TICKET
===========================

Ticket ID: MC-{datetime.now().strftime('%Y%m%d%H%M%S')}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Priority: MEDIUM
Category: User Query Support

USER INFORMATION:
----------------
User Email: {user_email if user_email else 'Not provided'}
Platform: MediChat Pro Web Application
Session Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

USER QUERY:
-----------
{user_query}

AI RESPONSE:
------------
{ai_response}
    """
    
    if chat_history and len(chat_history) > 0:
        text += "\n\nCHAT HISTORY CONTEXT:\n-------------------\n"
        for message in chat_history[-10:]:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            text += f"{role.upper()} ({timestamp}): {content}\n"
    
    text += """
SUPPORT NOTES:
--------------
Action Required: Please review the user query and AI response above.
Context: This ticket was automatically generated from a user's chat interaction with MediChat Pro.
Next Steps: Contact the user if additional information is needed or if the AI response requires human review.

---
🎫 MediChat Pro Support System | 🤖 Automated Ticket Generation
This support ticket was automatically generated by the MediChat Pro application
    """
    
    return text


def send_support_ticket(user_query: str, 
                       ai_response: str, 
                       user_email: str = None,
                       chat_history: List[Dict] = None) -> bool:
    """Send support ticket email"""
    try:
        message = create_support_ticket_email(
            user_query, ai_response, user_email, chat_history
        )
        return send_email(message)
    except Exception as e:
        st.error(f"Failed to create or send support ticket: {str(e)}")
        return False


def generate_document_insights(vectorstore, user_query: str, ai_response: str, 
                             relevant_docs: List, response_time: float) -> Dict[str, Any]:
    """Generate insights from document analysis"""
    
    # Extract medical keywords (basic implementation)
    medical_keywords = []
    medical_terms = [
        'diagnosis', 'treatment', 'medication', 'symptom', 'condition', 'disease',
        'therapy', 'prescription', 'allergy', 'blood pressure', 'heart rate',
        'temperature', 'pain', 'inflammation', 'infection', 'chronic', 'acute',
        'patient', 'doctor', 'nurse', 'hospital', 'clinic', 'medical', 'health'
    ]
    
    query_lower = user_query.lower()
    response_lower = ai_response.lower()
    
    for term in medical_terms:
        if term in query_lower or term in response_lower:
            medical_keywords.append(term)
    
    # Calculate confidence score based on response length and medical keywords
    confidence_score = min(100, (len(medical_keywords) * 10) + (len(ai_response) / 10))
    
    # Determine query complexity
    query_complexity = "Simple"
    if len(user_query.split()) > 10:
        query_complexity = "Complex"
    elif len(user_query.split()) > 5:
        query_complexity = "Medium"
    
    # Document coverage analysis
    doc_coverage = f"Analysis covered {len(relevant_docs)} relevant document chunks"
    if len(relevant_docs) > 3:
        doc_coverage += " with comprehensive coverage."
    elif len(relevant_docs) > 1:
        doc_coverage += " with moderate coverage."
    else:
        doc_coverage += " with limited coverage."
    
    return {
        'total_documents': 'Multiple',  # This would need to be tracked from session
        'total_chunks': len(relevant_docs) * 2,  # Estimate
        'relevant_docs_count': len(relevant_docs),
        'confidence_score': f"{confidence_score:.1f}%",
        'response_time': f"{response_time:.2f}",
        'query_complexity': query_complexity,
        'document_coverage': doc_coverage,
        'medical_keywords': medical_keywords[:10]  # Limit to top 10
    }


