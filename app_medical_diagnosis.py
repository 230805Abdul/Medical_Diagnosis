import streamlit as st
from pathlib import Path
import base64
import google.generativeai as genai
from api_key import api_key
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime


# Configure Gemini model
genai.configure(api_key=api_key)

# Generation config
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """You are a professional AI Medical Health Assistant. Your purpose is to provide clear, evidence-based general health information and predictive insights based on symptoms, medical history, and lifestyle factors shared by the user.

Your key responsibilities include:

Support, Not Diagnose:

You do not provide medical diagnoses or treatment plans.

You do help users better understand possible health conditions, explain medical concepts, and suggest appropriate next steps such as lifestyle adjustments or when to consult a healthcare professional.

Evidence-Based Guidance:

All information should be grounded in current, reputable clinical sources (e.g., CDC, WHO, Mayo Clinic, NICE guidelines).

Where relevant, cite standard clinical practices or widely accepted health recommendations.

Clear and Compassionate Communication:

Use language that is simple, respectful, and reassuring—never alarming.

If uncertainty exists, acknowledge it honestly and encourage the user to seek professional medical advice.

Always emphasize that no online tool can replace a qualified healthcare provider.

Privacy and Safety First:

Do not store, retain, or share any personal health information.

Avoid speculative or unsafe suggestions, even if prompted.

When in doubt, prioritize safety and recommend professional evaluation.

Scope and Boundaries:

Avoid making assumptions or offering unsupported conclusions.

Do not interpret lab results, imaging, or conduct risk assessments unless explicitly supported by clinical guidelines and generalizable data.

Example Disclaimer (to include in interactions):
“Please note: I am an AI health assistant and not a licensed medical professional. The information I provide is for general guidance only and should not be considered a medical diagnosis or a substitute for professional medical advice. If you are experiencing symptoms or have health concerns, please consult a licensed healthcare provider promptly.” """

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)


# SETUP & CONFIGURATION


# Page config
st.set_page_config(
    page_title="MediScan AI - Advanced Medical Diagnosis",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css(r"C:\Users\MR COMPUTER\Desktop\My Projects\medical_detection_app\style.css")


# SESSION STATE & NAVIGATION


if "page" not in st.session_state:
    st.session_state.page = "Home"


# PAGE COMPONENTS


def sidebar():
    with st.sidebar:
        st.image(r"C:\Users\MR COMPUTER\Desktop\My Projects\medical_detection_app\Screenshot 2025-04-28 224724.png", use_container_width=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-header">Navigation</h3>
            <ul class="sidebar-menu">
        """, unsafe_allow_html=True)
        
        menu_items = {
            "Home": "",
            "AI Diagnosis": "",
            "Health Insights": "",
            "Disease Encyclopedia": "",
            "Prevention Hub": "",
            "Risk Assessment": "",
            "Medical Resources": "",
            "FAQ": "",
            "Contact": ""
        }
        
        for item, icon in menu_items.items():
            if st.button(f"{icon} {item}", key=f"menu_{item}"):
                st.session_state.page = item
            st.markdown("<style>div[data-testid='stButton'] > button {width: 100%;}</style>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Add user section
        st.markdown("""
        <div class="sidebar-section user-section">
            <h3 class="sidebar-header">Your Health Profile</h3>
            <div class="user-avatar">👤</div>
            <p class="user-name">Welcome, User</p>
        </div>
        """, unsafe_allow_html=True)


# PAGE LAYOUTS


def home_page():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Revolutionizing Healthcare with AI</h1>
            <p class="hero-subtitle">Your Personal Health Companion for Accurate, Instant Medical Insights</p>
            <button class="primary-button">Get Started</button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3> Instant Analysis</h3>
            <p>Upload medical images or reports and receive AI-powered insights in seconds.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.image(r"C:\Users\MR COMPUTER\Desktop\My Projects\medical_detection_app\Screenshot 2025-04-28 224724.png", width = 300)
    
    st.markdown("""
    <div class="features-section">
        <h2 class="section-title">Why Choose MediScan AI?</h2>
        <div class="features-grid">
            <div class="feature-item">
                <div class="feature-icon">⚡</div>
                <h4>Lightning Fast</h4>
                <p>Get results in minutes, not days</p>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🎯</div>
                <h4>Highly Accurate</h4>
                <p>Powered by advanced AI models</p>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔒</div>
                <h4>Secure & Private</h4>
                <p>Your data stays confidential</p>
            </div>
            <div class="feature-item">
                <div class="feature-icon">💡</div>
                <h4>Actionable Insights</h4>
                <p>Clear next steps for your health</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def diagnosis_page():
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">AI-Powered Medical Diagnosis</h1>
        <p class="page-subtitle">Upload your medical images or reports for instant analysis</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Report Analysis", "Symptom Checker"])

    # --- Tab 1: Image Analysis ---
    with tab1:
        st.markdown("""
        <div class="upload-card">
            <h3>Upload Medical Image</h3>
            <p>Supported formats: JPG, PNG, DICOM</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a medical image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                image = Image.open(uploaded_file)

            with col2:
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Analyzing image with AI..."):
                        try:
                            # Prepare the image for Gemini
                            img_prompt = """
                            You are a medical imaging specialist analyzing this image. Provide:
                            1. A professional assessment of any visible abnormalities
                            2. Potential conditions that could explain these findings
                            3. Recommended next steps (imaging follow-up, specialist consultation)
                            4. Urgency level (routine, moderate, urgent)
                            
                            Be factual but compassionate. Always remind this is not a diagnosis.
                            """
                            
                            # Call Gemini with the image
                            response = model.generate_content([img_prompt, image])
                            
                            # Display results
                            st.success("Analysis complete!")
                            st.markdown(f"""
                            <div class="result-card">
                                <h3>AI Analysis Results</h3>
                                <div class="result-content">
                                    {response.text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.info("Please try with a different image or consult a healthcare provider")

    # --- Tab 2: Report Analysis ---
    with tab2:
        st.markdown("""
        <div class="upload-card">
            <h3>Upload Lab Report</h3>
            <p>Supported formats: PDF, TXT, CSV</p>
        </div>
        """, unsafe_allow_html=True)

        report_file = st.file_uploader("Choose a report file", type=["pdf", "txt", "csv"], key="report_uploader", label_visibility="collapsed")

        if report_file:
            if st.button("Analyze Report", type="primary"):
                with st.spinner("Processing report..."):
                    try:
                        # Read the report content
                        if report_file.type == "application/pdf":
                            # For PDFs we'd need additional libraries like PyPDF2
                            report_text = "[PDF content extracted would appear here]"
                        else:
                            report_text = report_file.getvalue().decode("utf-8")
                        
                        # Prepare the report prompt
                        report_prompt = f"""
                        Analyze this medical report and provide:
                        1. Summary of key abnormal findings
                        2. Potential health implications
                        3. Recommended follow-up actions
                        4. General health advice
                        
                        Report content:
                        {report_text[:3000]}... [truncated if long]
                        """
                        
                        # Call Gemini with the report text
                        response = model.generate_content(report_prompt)
                        
                        st.success("Report analysis complete!")
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>Report Analysis Summary</h3>
                            <div class="result-content">
                                {response.text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Report analysis failed: {str(e)}")

    # --- Tab 3: Symptom Checker ---
    with tab3:
        st.markdown("""
        <div class="symptom-card">
            <h3>Check Your Symptoms</h3>
            <p>Select the symptoms you're experiencing below.</p>
        </div>
        """, unsafe_allow_html=True)

        symptoms = st.multiselect(
            "Choose your symptoms:",
            options=[
                "Fever", "Cough", "Chest Pain", "Shortness of Breath", "Fatigue",
                "Nausea", "Headache", "Abdominal Pain", "Joint Pain", "Skin Rash"
            ],
            label_visibility="collapsed"
        )

        duration = st.selectbox("Duration of symptoms", ["Less than 24 hours", "1-3 days", "3-7 days", "1-2 weeks", "More than 2 weeks"])
        severity = st.select_slider("Symptom severity", ["Mild", "Moderate", "Severe"])

        if st.button("Check Possible Conditions"):
            if not symptoms:
                st.warning("Please select at least one symptom.")
            else:
                with st.spinner("Analyzing symptoms..."):
                    try:
                        symptom_prompt = f"""
                        A user reports these symptoms:
                        - Main symptoms: {", ".join(symptoms)}
                        - Duration: {duration}
                        - Severity: {severity}
                        
                        Provide:
                        1. 2-3 most likely general conditions (not diagnoses)
                        2. Recommended self-care measures
                        3. When to seek medical attention
                        4. Red flag symptoms to watch for
                        
                        Be conservative and always recommend professional evaluation when uncertain.
                        """
                        
                        response = model.generate_content(symptom_prompt)
                        
                        st.success("Preliminary Insight:")
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>Symptom Analysis</h3>
                            <div class="result-content">
                                {response.text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Symptom analysis failed: {str(e)}")

def health_insights_page():
    if 'user_health_data' not in st.session_state:
        st.session_state.user_health_data = pd.DataFrame(columns=['Date', 'Blood Pressure', 'Cholesterol', 'Heart Rate'])
    st.title("Health Insights")
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Your Health Insights Dashboard</h1>
        <p class="page-subtitle">Track and analyze your health metrics over time</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###  Enter Your Health Metrics")
    
    with st.form("health_input_form"):
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=75)
        submit = st.form_submit_button("Add Entry")
        
        if submit:
            new_data = pd.DataFrame([{
                'Date': datetime.today().date(),
                'Blood Pressure': bp,
                'Cholesterol': cholesterol,
                'Heart Rate': heart_rate
            }])
            st.session_state.user_health_data = pd.concat([st.session_state.user_health_data, new_data], ignore_index=True)
            st.success("Health entry added!")

    # Visualizations
    if not st.session_state.user_health_data.empty:
        data = st.session_state.user_health_data

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Your Health Trends")
            tab1, tab2, tab3 = st.tabs(["Blood Pressure", "Cholesterol", "Heart Rate"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=data, x='Date', y='Blood Pressure', marker='o', ax=ax)
                ax.set_title("Blood Pressure Over Time")
                st.pyplot(fig)
                
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=data, x='Date', y='Cholesterol', marker='o', color='orange', ax=ax)
                ax.set_title("Cholesterol Over Time")
                st.pyplot(fig)
                
            with tab3:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=data, x='Date', y='Heart Rate', marker='o', color='green', ax=ax)
                ax.set_title("Heart Rate Over Time")
                st.pyplot(fig)
        
        with col2:
            st.markdown("### 💡 AI Health Summary")
            latest = data.iloc[-1]
            if latest['Blood Pressure'] < 130:
                st.success("✅ Blood Pressure is in a healthy range.")
            else:
                st.warning("⚠️ Monitor your blood pressure.")

            if latest['Cholesterol'] > 200:
                st.warning("⚠️ Cholesterol is above the recommended level.")
            else:
                st.success("✅ Cholesterol is in a healthy range.")

            if 60 <= latest['Heart Rate'] <= 100:
                st.success("✅ Heart rate is normal.")
            else:
                st.warning("⚠️ Abnormal heart rate detected.")
    else:
        st.info("No data to display yet. Please enter your health metrics above.")

def disease_insights():
    st.title(" Disease Insights")
    disease = st.selectbox("Select a condition to learn more:", ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "COVID-19"])
    
    disease_info = {
        "Diabetes": "A chronic condition affecting how your body turns food into energy. Management involves lifestyle changes and possibly medication.",
        "Hypertension": "High blood pressure often has no symptoms but can lead to serious health issues. Regular monitoring and healthy living are key.",
        "Asthma": "A respiratory condition marked by spasms in the bronchi of the lungs, causing difficulty in breathing.",
        "Heart Disease": "Includes conditions like coronary artery disease, heart attacks, and arrhythmias. It’s the leading cause of death globally.",
        "COVID-19": "A viral respiratory illness caused by SARS-CoV-2. Preventive measures and vaccination reduce risk of severe outcomes."
    }

    st.info(disease_info[disease])

def prevention_hub():
    st.title(" Prevention Hub")
    st.markdown("""
    ### General Preventive Tips
    -  Eat a balanced, nutrient-rich diet
    -  Exercise at least 30 minutes a day
    -  Avoid tobacco and limit alcohol
    -  Sleep 7-9 hours each night
    -  Wash hands regularly and practice hygiene
    -  Stay up to date with vaccinations
    -  Manage stress through mindfulness or meditation
    """)

def risk_assessment():
    st.title(" Health Risk Assessment")

    age = st.slider("Your Age", 10, 90, 30)
    bmi = st.number_input("Your BMI", min_value=10.0, max_value=50.0)
    smoker = st.radio("Do you smoke?", ["No", "Yes"])
    activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    
    risk_score = 0
    risk_score += 1 if age > 50 else 0
    risk_score += 1 if bmi >= 30 else 0
    risk_score += 1 if smoker == "Yes" else 0
    risk_score += 1 if activity == "Low" else 0

    if st.button("Assess Risk"):
        if risk_score <= 1:
            st.success("✅ Your health risk is Low. Keep up the healthy habits!")
        elif risk_score == 2:
            st.warning("⚠️ Moderate risk. Consider lifestyle improvements.")
        else:
            st.error("❗High risk. Please consult a healthcare provider.")

def medical_resources():
    st.title(" Medical Resources")

    st.markdown("""
    ### Trusted Health Websites
    - [World Health Organization (WHO)](https://www.who.int/)
    - [Centers for Disease Control and Prevention (CDC)](https://www.cdc.gov/)
    - [Mayo Clinic](https://www.mayoclinic.org/)
    - [WebMD](https://www.webmd.com/)
    - [National Institutes of Health (NIH)](https://www.nih.gov/)
    
    ### Medical Hotline Numbers (Country-Specific)
    - Emergency: 112 / 911
    - COVID-19 Helpline: [Local Ministry of Health site]
    """)

def faq_section():
    st.title(" Frequently Asked Questions")

    with st.expander("Is this app a substitute for a doctor?"):
        st.info("No. This app provides general health insights and does not replace professional medical advice.")

    with st.expander("Is my data stored?"):
        st.info("No. All data is processed in-session and not stored permanently.")

    with st.expander("Can I get a prescription?"):
        st.warning("No. Only a licensed physician can issue medical prescriptions.")

    with st.expander("Which files can I upload?"):
        st.info("Supported files include JPG, PNG, DICOM for images and PDF, TXT, CSV for reports.")

def contact_page():
    st.title(" Contact & Feedback")

    st.markdown("""
    We'd love to hear from you. For support, questions, or suggestions:
    
    -  **Email**: abdullbasit0023@gmail.com
    -  **Website**: [www.aimedicalassist.com](#)
    -  **Feedback Form**: [Fill here](#)
    -  **Support Hours**: Mon-Fri, 9AM - 5PM (Local Time)
    """)



# MAIN APP LOGIC


def main():
    sidebar()
    
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "AI Diagnosis":
        diagnosis_page()
    elif st.session_state.page == "Health Insights":
        health_insights_page()
    elif st.session_state.page == "FAQ":
        faq_section()
    elif st.session_state.page == "Contact":
        contact_page()
    elif st.session_state.page == "Medical Resources":
        medical_resources()
    elif st.session_state.page == "Risk Assessment":
        risk_assessment()
    elif st.session_state.page == "Prevention Hub":
        prevention_hub()
    elif st.session_state.page == "Disease Encyclopedia":
        disease_insights()
    

if __name__ == "__main__":
    main()
