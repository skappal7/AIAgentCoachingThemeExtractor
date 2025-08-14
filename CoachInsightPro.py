import streamlit as st
import pandas as pd
import requests
import json
import io
from typing import List, Dict, Any
import time
import re
from datetime import datetime

# Try to import chardet, fallback if not available
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    st.warning("chardet library not available. Using UTF-8 as default encoding.")

# Page configuration
st.set_page_config(
    page_title="CoachInsight Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .upload-section {
        background: #f8f9ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .analysis-result {
        background: linear-gradient(135deg, #f5f7ff 0%, #e8ecff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .skill-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "moonshotai/kimi-k2:free"

def detect_encoding(file_content: bytes) -> str:
    """Detect file encoding using chardet if available, otherwise use common encodings"""
    if CHARDET_AVAILABLE:
        result = chardet.detect(file_content)
        return result['encoding'] or 'utf-8'
    else:
        # Fallback: try common encodings
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings_to_try:
            try:
                file_content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # Final fallback

def read_file_with_encoding(uploaded_file) -> pd.DataFrame:
    """Read uploaded file with proper encoding detection"""
    try:
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        if uploaded_file.name.endswith('.csv'):
            encoding = detect_encoding(file_content)
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
            except UnicodeDecodeError:
                # Fallback to utf-8 with error handling
                df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8', errors='replace')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
        elif uploaded_file.name.endswith('.txt'):
            encoding = detect_encoding(file_content)
            try:
                content = file_content.decode(encoding)
            except UnicodeDecodeError:
                content = file_content.decode('utf-8', errors='replace')
            # Convert text file to dataframe with each line as a row
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({'feedback': lines})
        else:
            st.error("Unsupported file format!")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def call_ai_model(prompt: str, max_retries: int = 3) -> str:
    """Call the AI model with retry logic"""
    # Validate API key format
    if not API_KEY or not API_KEY.startswith('sk-or-v1-'):
        st.error("❌ Invalid API key format! OpenRouter keys should start with 'sk-or-v1-'")
        return None
        
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://coachinsight-pro.streamlit.app",
                "X-Title": "CoachInsight Pro"
            }
            
            data = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert in analyzing agent coaching feedback and identifying training opportunities. 
                        Your task is to extract specific training themes and skill-related keywords from coaching feedback text.
                        Always respond in JSON format with 'training_themes' (specific actionable training topics) and 'skill_keywords' (individual skills/competencies)."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(API_URL, headers=headers, data=json.dumps(data), timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 401:
                st.error("🚨 **Authentication Error (401)**: Invalid API key or account not found.")
                st.error("Please check:")
                st.error("• API key is correct and valid")
                st.error("• Account exists on OpenRouter")
                st.error("• API key has proper permissions")
                st.info("💡 **How to get a valid API key:**")
                st.info("1. Go to https://openrouter.ai/")
                st.info("2. Sign up or log in")
                st.info("3. Go to API Keys section")
                st.info("4. Generate a new API key")
                return None
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                st.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 403:
                st.error("🚨 **Access Forbidden (403)**: Model access denied.")
                st.error("This could mean:")
                st.error("• The free tier limit has been reached")
                st.error("• Model access is restricted")
                st.error("• Account verification required")
                return None
            else:
                st.error(f"🚨 **API Error {response.status_code}**: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Request timeout. Attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
        except Exception as e:
            st.error(f"❌ Error calling AI model: {str(e)}")
            return None
    
    return None

def extract_training_insights(feedback_text: str) -> Dict[str, Any]:
    """Extract training themes and skills from feedback text"""
    prompt = f"""
    Analyze this agent coaching feedback and extract training opportunities:
    
    "{feedback_text}"
    
    Please identify:
    1. Specific training themes (actionable training topics like "Active Listening Training", "Objection Handling", etc.)
    2. Individual skill keywords (core competencies like "communication", "empathy", "product knowledge", etc.)
    
    Respond in this exact JSON format:
    {{
        "training_themes": ["Training Topic 1", "Training Topic 2"],
        "skill_keywords": ["skill1", "skill2", "skill3"]
    }}
    """
    
    response = call_ai_model(prompt)
    if not response:
        return {"training_themes": [], "skill_keywords": []}
    
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # Fallback parsing
            return {"training_themes": [], "skill_keywords": []}
    except json.JSONDecodeError:
        return {"training_themes": [], "skill_keywords": []}

def process_feedback_data(df: pd.DataFrame, feedback_column: str) -> pd.DataFrame:
    """Process the feedback data and add AI insights"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Processing feedback {idx + 1} of {len(df)}...")
        progress_bar.progress((idx + 1) / len(df))
        
        feedback_text = str(row[feedback_column])
        if pd.isna(feedback_text) or feedback_text.strip() == "":
            insights = {"training_themes": [], "skill_keywords": []}
        else:
            insights = extract_training_insights(feedback_text)
        
        result_row = row.to_dict()
        result_row['training_themes'] = ', '.join(insights.get('training_themes', []))
        result_row['skill_keywords'] = ', '.join(insights.get('skill_keywords', []))
        
        results.append(result_row)
        
        # Add delay to respect rate limits
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 CoachInsight Pro</h1>
        <p>Intelligent Agent Coaching Analysis & Training Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "OpenRouter API Key",
            value="",
            type="password",
            help="Enter your OpenRouter API key (starts with sk-or-v1-)"
        )
        
        if api_key_input:
            global API_KEY
            API_KEY = api_key_input
            st.success("✅ API Key configured!")
        elif not API_KEY or API_KEY == "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":
            st.warning("⚠️ Please enter your OpenRouter API key")
            st.info("Get your free API key at: https://openrouter.ai/")
        
        # Test API connection
        if st.button("🧪 Test API Connection"):
            if API_KEY and API_KEY.startswith('sk-or-v1-'):
                with st.spinner("Testing API connection..."):
                    test_result = call_ai_model("Hello, please respond with 'API connection successful'")
                    if test_result:
                        st.success("✅ API connection successful!")
                    else:
                        st.error("❌ API connection failed!")
            else:
                st.error("❌ Invalid API key format!")
        
        st.markdown("---")
        
        st.markdown("### 📊 Analysis Dashboard")
        st.markdown("---")
        
        st.markdown("#### ℹ️ How it works:")
        st.markdown("""
        1. **Upload** your coaching feedback file
        2. **Select** the feedback column
        3. **Analyze** with AI-powered insights
        4. **Download** results with training themes
        """)
        
        st.markdown("---")
        st.markdown("#### 📁 Supported Formats:")
        st.markdown("• CSV files (.csv)")
        st.markdown("• Excel files (.xlsx, .xls)")
        st.markdown("• Text files (.txt)")
        
        st.markdown("---")
        st.markdown("#### 🔍 Analysis Features:")
        st.markdown("• Training theme identification")
        st.markdown("• Skill gap analysis")
        st.markdown("• Actionable insights")
        st.markdown("• Automated categorization")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📤 Upload Your Coaching Feedback Data</h3>
            <p>Upload CSV, Excel, or Text files containing agent coaching feedback</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your file",
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="Upload files with agent coaching feedback data"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Reading and processing file..."):
                df = read_file_with_encoding(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded successfully! Found {len(df)} records")
                
                # Display file info
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column selection
                st.markdown("### 🎯 Select Feedback Column")
                feedback_column = st.selectbox(
                    "Choose the column containing coaching feedback:",
                    df.columns.tolist(),
                    help="Select the column that contains the coaching feedback text"
                )
                
                if st.button("🚀 Start Analysis", type="primary"):
                    with st.spinner("🤖 AI is analyzing your coaching feedback..."):
                        try:
                            results_df = process_feedback_data(df, feedback_column)
                            
                            st.success("✅ Analysis completed!")
                            
                            # Display results
                            st.markdown("### 📊 Analysis Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary metrics
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.markdown("""
                                <div class="metric-card">
                                    <h4>📈 Records Processed</h4>
                                    <h2>{}</h2>
                                </div>
                                """.format(len(results_df)), unsafe_allow_html=True)
                            
                            with col_b:
                                training_themes = results_df['training_themes'].str.split(', ').explode()
                                unique_themes = len([t for t in training_themes.unique() if t and t.strip()])
                                st.markdown("""
                                <div class="metric-card">
                                    <h4>🎯 Training Themes</h4>
                                    <h2>{}</h2>
                                </div>
                                """.format(unique_themes), unsafe_allow_html=True)
                            
                            with col_c:
                                skill_keywords = results_df['skill_keywords'].str.split(', ').explode()
                                unique_skills = len([s for s in skill_keywords.unique() if s and s.strip()])
                                st.markdown("""
                                <div class="metric-card">
                                    <h4>💡 Skill Keywords</h4>
                                    <h2>{}</h2>
                                </div>
                                """.format(unique_skills), unsafe_allow_html=True)
                            
                            # Top themes and skills
                            st.markdown("### 📈 Top Training Themes")
                            all_themes = []
                            for themes_str in results_df['training_themes']:
                                if themes_str and themes_str.strip():
                                    all_themes.extend([t.strip() for t in themes_str.split(',') if t.strip()])
                            
                            if all_themes:
                                theme_counts = pd.Series(all_themes).value_counts().head(10)
                                st.bar_chart(theme_counts)
                            
                            st.markdown("### 🏷️ Top Skill Keywords")
                            all_skills = []
                            for skills_str in results_df['skill_keywords']:
                                if skills_str and skills_str.strip():
                                    all_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
                            
                            if all_skills:
                                skill_counts = pd.Series(all_skills).value_counts().head(10)
                                st.bar_chart(skill_counts)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Analysis Results",
                                data=csv,
                                file_name=f"coaching_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                type="primary"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="analysis-result">
            <h4>🎯 What You'll Get</h4>
            <ul>
                <li><strong>Training Themes:</strong> Specific actionable training recommendations</li>
                <li><strong>Skill Keywords:</strong> Individual competencies and skills identified</li>
                <li><strong>Comprehensive Analysis:</strong> AI-powered insights for each feedback entry</li>
                <li><strong>Downloadable Results:</strong> CSV export with all original data plus insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="analysis-result">
            <h4>💡 Sample Output</h4>
            <div class="skill-tag">Active Listening</div>
            <div class="skill-tag">Product Knowledge</div>
            <div class="skill-tag">Objection Handling</div>
            <div class="skill-tag">Communication Skills</div>
            <div class="skill-tag">Empathy Training</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
