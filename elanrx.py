import streamlit as st
from knowledge_base import get_response
from model import ChestXRayAnalyzer
from PIL import Image
import os
import time

# Initialize analyzer
analyzer = ChestXRayAnalyzer()

# Set up the app
st.set_page_config(page_title="Chest X-ray Chatbot", page_icon="🩺")

# App header
st.title("🩺 Chest X-ray Chatbot")
st.markdown("Upload a chest X-ray or ask questions about common findings")

# Create tabs
tab1, tab2 = st.tabs(["Chat", "Image Analysis"])

# Chat tab
with tab1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about chest X-rays..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        response = get_response(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Image Analysis tab
with tab2:
    st.header("Upload Chest X-ray for Analysis")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Create uploads directory if not exists
        os.makedirs("assets/uploads", exist_ok=True)

        # Save uploaded file
        upload_path = os.path.join("assets/uploads", uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display image
        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

        # Analyze button
        if st.button("Analyze X-ray"):
            with st.spinner("Analyzing image..."):
                # Get analysis results
                results = analyzer.analyze_image(upload_path)
                time.sleep(2)  # Simulate processing time

                # Display results
                st.subheader("Analysis Results")

                # Show confidence bars
                for condition, prob in results.items():
                    st.write(f"{condition}: {prob*100:.2f}%")
                    st.progress(prob)

                # Get the most likely condition
                primary_condition = max(results, key=results.get)
                st.info(f"Primary finding: {primary_condition}")

                # Show information about the condition
                st.subheader(f"About {primary_condition}")
                st.markdown(get_response(f"what is {primary_condition.lower()}?"))

# Sidebar
st.sidebar.markdown("### Quick Links")
st.sidebar.markdown("- [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)")
st.sidebar.markdown("- [Radiology Guidelines](https://www.acr.org/Clinical-Resources/Radiology-Topics)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Disclaimer**: This tool provides AI-assisted analysis but is not a substitute for professional medical diagnosis.")