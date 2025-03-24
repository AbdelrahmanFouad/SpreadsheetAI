import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
import google.generativeai as genai
import re
import io

st.set_page_config(layout="wide", page_title="excel magic")
# ✅ Configure Google Gemini API Key
genai.configure(api_key="")  # Replace with your API key


# Function to remove ANSI escape sequences from text
def remove_ansi_escape(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# ✅ Custom Google Gemini LLM
class GoogleGeminiLLM(LLM):
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key

    def call(self, prompt: str, context=None, **kwargs) -> str:
        """Sends a plain-text prompt to Gemini and ensures it returns clean Python code."""
        model = genai.GenerativeModel(self.model_name)
        prompt_text = (
            f"{prompt}\n\n"
            "Return only valid Python code with no explanations, no JSON, and no Markdown formatting."
            "Ensure the output is directly executable without additional context."
            "The code you generate will be used for pandasai."
        )
        response = model.generate_content(prompt_text)
        raw_code = response.text if response else "print('No response from Gemini')"
        clean_code = re.sub(r"```(?:python)?\n(.*?)```", r"\1", raw_code, flags=re.DOTALL).strip()
        clean_code = remove_ansi_escape(clean_code)
        return clean_code

    def generate_code(self, prompt, context=None):
        return self.call(str(prompt), context)

    @property
    def type(self):
        return "Google Gemini"

# ✅ Initialize Gemini LLM
gemini_llm = GoogleGeminiLLM(api_key="AIzaSyCPM4iPRO1FrL1SduszTktoR_W1huGu1TA")  # Replace with your API key

st.title("I hate excel")
st.write("Upload a CSV or Excel file, enter your transformation prompt, and preview/download the modified data.")

# File uploader for CSV or Excel
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        df.columns = df.columns.str.replace(r'\W+', '_', regex=True)  
        # Clean column names from ANSI escape sequences
        df.columns = [remove_ansi_escape(col) for col in df.columns]
        st.subheader("Original Data")
        st.dataframe(df)
        
        # Prompt input for data transformation
        prompt = st.text_area(
            "Enter your prompt for data transformation",
            "ex: remove second column"
        )
        
        if st.button("Process Data"):
            # Wrap DataFrame with PandasAI SmartDataframe
            sdf = SmartDataframe(df, config={"llm": gemini_llm})
            try:
                response = sdf.chat(prompt)
                
                # Check if response is already a pandas DataFrame,
                # otherwise try to get the underlying DataFrame from the 'value' attribute.
                if isinstance(response, pd.DataFrame):
                    modified_df = response
                elif hasattr(response, "value"):
                    modified_df = response.value
                else:
                    st.error("The response from PandasAI could not be converted to a DataFrame.")
                    modified_df = None
                
                if modified_df is not None:
                    st.subheader("Transformed Data")
                    st.dataframe(modified_df)
                    
                    # Download options as separate buttons using columns
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = modified_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="modified_data.csv",
                            mime="text/csv"
                        )
                    with col2:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            modified_df.to_excel(writer, index=False, sheet_name='Sheet1')
                        output.seek(0)  # Reset pointer to beginning of the buffer
                        processed_data = output.getvalue()
                        st.download_button(
                            label="Download Excel",
                            data=processed_data,
                            file_name="modified_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.subheader("Response from Gemini")
                    st.code(response, language="python")
            except Exception as e:
                st.error("Error processing prompt: " + str(e))
    except Exception as e:
        st.error("Error loading file: " + str(e))
