import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

class DocumentClassifierUI:
    def __init__(self):
        self.API_URL = "http://localhost:8000"
        if 'history' not in st.session_state:
            st.session_state.history = []

    def create_confidence_chart(self, result):
        """Create confidence score visualization."""
        if not result.get('possible_types'):
            return None
        
        df = pd.DataFrame(result['possible_types'], columns=['Document Type', 'Confidence'])
        
        fig = px.bar(
            df,
            x='Document Type',
            y='Confidence',
            title='Classification Confidence Scores',
            color='Confidence',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Document Type",
            yaxis_title="Confidence Score",
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig

    def process_file(self, file):
        """Process a single file."""
        try:
            files = {"file": (file.name, file, file.type)}
            response = requests.post(f"{self.API_URL}/classify-webapp", files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result:
                    # Store processing time in metadata if not present
                    if 'processing_time' not in result.get('metadata', {}):
                        result['metadata']['processing_time'] = 0.0
                return result
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

    def display_result(self, result):
        """Display classification result."""
        if not result:
            return

        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Result")
            st.json({
                "document_type": result["document_type"],
                "confidence_score": f"{result['confidence_score']:.2%}",
                "processing_time": f"{result.get('metadata', {}).get('processing_time', 0):.2f}s"
            })
            
            # Display metadata
            st.subheader("Document Metadata")
            # Filter out processing_time from metadata display
            metadata_display = {
                k: v for k, v in result.get('metadata', {}).items()
                if k != 'processing_time'
            }
            st.json(metadata_display)

        with col2:
            st.subheader("Confidence Analysis")
            fig = self.create_confidence_chart(result)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    def display_history(self):
        """Display processing history."""
        if st.session_state.history:
            st.subheader("Processing History")
            history_df = pd.DataFrame([
                {
                    'Time': entry['timestamp'],
                    'Filename': entry['filename'],
                    'Type': entry['document_type'],
                    'Confidence': f"{entry['confidence_score']:.2%}",
                    'Processing Time': f"{entry.get('metadata', {}).get('processing_time', 0):.2f}s"
                }
                for entry in st.session_state.history
            ])
            st.dataframe(history_df)

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(page_title="Document Classifier", layout="wide")
        
        st.title("📄 Document Classification System")
        st.write("Upload documents to automatically classify them.")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Supported formats: PDF, PNG, JPG"
        )

        if uploaded_file:
            with st.spinner("Processing document..."):
                result = self.process_file(uploaded_file)
                
                if result:
                    st.success("Document processed successfully!")
                    self.display_result(result)
                    
                    # Add to history
                    st.session_state.history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        **result
                    })

        # Display history
        self.display_history()

        # Sidebar
        with st.sidebar:
            st.header("System Status")
            try:
                response = requests.get(f"{self.API_URL}/health")
                if response.status_code == 200:
                    st.success("API Status: Online")
                else:
                    st.error("API Status: Offline")
            except:
                st.error("API Status: Cannot connect to API")

            st.header("Supported Document Types")
            st.markdown("""
            - Driver's Licenses
            - Bank Statements
            - Invoices
            - Payslips
            - Tax Returns
            - Utility Bills
            """)

if __name__ == "__main__":
    app = DocumentClassifierUI()
    app.run()
