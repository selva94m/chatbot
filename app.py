import streamlit as st
import pandas as pd
from dynamic_servicenow_bot import DynamicServiceNowBot
import os

# Set page configuration
st.set_page_config(
    page_title="Azure DevOps Support Bot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'bot' not in st.session_state:
    # Check if data file exists
    data_file = "azure_devops_servicenow_sample.xlsx"
    if not os.path.exists(data_file):
        st.session_state.data_generated = False
    else:
        st.session_state.data_generated = True
        st.session_state.bot = DynamicServiceNowBot(data_file)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("Azure DevOps Support Bot")
st.markdown("""
This bot provides solutions to common Azure DevOps issues based on ServiceNow ticket data.
Ask any question related to Azure DevOps and get instant solutions!
""")

# Sidebar for settings and data management
with st.sidebar:
    st.header("Bot Settings")
    
    threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.3,
        step=0.05,
        help="Minimum similarity score to consider a match valid"
    )
    
    st.header("Data Management")
    
    # Data generation or upload
    if not st.session_state.get('data_generated', False):
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                bot = DynamicServiceNowBot("azure_devops_servicenow_sample.xlsx")
                st.session_state.bot = bot
                st.session_state.data_generated = True
                st.success("Sample data generated successfully!")
                st.rerun()
    else:
        st.success("✅ Bot is ready with sample data")
        
        # Option to view the data
        if st.button("View Knowledge Base"):
            data = st.session_state.bot.data
            st.dataframe(data[['question', 'solution']])
        
        # Option to upload custom data
        uploaded_file = st.file_uploader("Upload custom data (Excel)", type=["xlsx"])
        if uploaded_file is not None:
            try:
                with st.spinner("Loading custom data..."):
                    custom_data = pd.read_excel(uploaded_file)
                    if 'question' in custom_data.columns and 'solution' in custom_data.columns:
                        # Save the uploaded file
                        custom_data.to_excel("custom_data.xlsx", index=False)
                        # Reinitialize the bot with the new data
                        st.session_state.bot = DynamicServiceNowBot("custom_data.xlsx")
                        st.success("Custom data loaded successfully!")
                    else:
                        st.error("The uploaded file must contain 'question' and 'solution' columns.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

# Main chat interface
st.header("Ask your question")

# Input box for query
query = st.text_input("Type your Azure DevOps question here:", key="query_input")

# Process the query when submitted
if query:
    if not st.session_state.get('data_generated', False):
        st.warning("Please generate or upload data first using the sidebar options.")
    else:
        with st.spinner("Finding solutions..."):
            # Get solution from bot
            result = st.session_state.bot.get_solution(query, threshold=threshold)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "result": result
            })
            
            # Clear the input box by changing the key
            st.text_input("Type your Azure DevOps question here:", key=f"query_input_{len(st.session_state.chat_history)}")

# Display chat history
st.header("Solutions")
for i, chat in enumerate(reversed(st.session_state.chat_history)):
    result = chat["result"]
    
    # Create expandable section for each Q&A
    with st.expander(f"Question: {chat['query']}", expanded=(i == 0)):
        if result['matched_question']:
            st.markdown(f"**Matched question:** {result['matched_question']}")
            st.progress(result['confidence'])
            st.markdown(f"**Confidence score:** {result['confidence']:.2f}")
            
            # Display the solution with proper formatting
            st.markdown("### Solution:")
            st.markdown(result['solution'])
            
            # Show alternatives if available
            if result['alternatives']:
                st.markdown("### Other possible solutions:")
                for j, alt in enumerate(result['alternatives'], 1):
                    st.markdown(f"**{j}. Related to:** {alt['question']} (Confidence: {alt['confidence']:.2f})")
                    st.markdown(alt['solution'])
        else:
            st.warning(result['solution'])

# Clear chat history button
if st.session_state.chat_history and st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
