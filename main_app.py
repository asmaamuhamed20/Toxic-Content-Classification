import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from image_caption import ImageCaptioner
from text_classifier import get_classifier
from database_manager import DatabaseManager

# Page configuration
st.set_page_config(page_title="Toxic Content Classification", page_icon=" ", layout="wide")

@st.cache_resource
def load_models():
    try:
        captioner = ImageCaptioner()
        classifier = get_classifier()
        db = DatabaseManager()
        return captioner, classifier, db
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    st.title("Toxic Content Classification System")
    
    captioner, classifier, db = load_models()
    
    if captioner is None:
        st.error("Failed to load models")
        return

    tab1, tab2, tab3 = st.tabs(["Text Classification", "Image Classification", "Database View"])
    
    with tab1:
        st.header("Text Content Classification")
        text_input = st.text_area("Enter text to classify:", placeholder="Type your text here...", height=150)
        
        if st.button("Classify Text", type="primary") and text_input.strip():
            with st.spinner("Analyzing text..."):
                classification, score = classifier.classify(text_input)
                
                col1, col2 = st.columns(2)
                with col1:
                    if classification == "toxic":
                        st.error(f"Toxic Content (Score: {score:.3f})")
                    else:
                        st.success(f" Safe Content (Score: {score:.3f})")
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Toxicity Score"}, gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "darkblue"}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                db.add_entry("text", text_input, "", classification, score)
                st.success("Result saved to database!")
    
    with tab2:
        st.header("Image Content Classification")
        uploaded_file = st.file_uploader("Choose an image:", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file and st.button("Analyze Image", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("Generating caption..."):
                    caption = captioner.generate_caption(tmp_path)
                    if caption:
                        st.subheader("Generated Caption:")
                        st.info(caption)
                        
                        classification, score = classifier.classify(caption)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if classification == "toxic":
                                st.error(f"Toxic Content (Score: {score:.3f})")
                            else:
                                st.success(f"Safe Content (Score: {score:.3f})")
                        
                        db.add_entry("image", uploaded_file.name, caption, classification, score)
                        st.success("Result saved to database!")
            finally:
                os.unlink(tmp_path)
    
    with tab3:
        st.header("Database View")
        df = db.get_all_entries()
        
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv_data, file_name="database.csv")
        else:
            st.info("No data yet")

if __name__ == "__main__":
    main()