import streamlit as st
import pandas as pd
import json
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
import google.generativeai as genai
import re
import io
import chardet  

st.set_page_config(layout="wide", page_title="Excelerate")

genai.configure(api_key=st.secrets["google"]["api_key"])

def remove_ansi_escape(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def clear_session_keys(keys):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def extract_metadata_and_sample(df, sample_size=5):
    metadata = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict()
    }
    sample = df.head(sample_size)
    return metadata, sample

def create_composite_prompt(user_prompt, metadata, sample):
    composite_prompt = (
        "Below is the metadata and a sample of the data:\n"
        f"Metadata:\n{json.dumps(metadata, indent=2)}\n\n"
        f"Sample Data (first {len(sample)} rows):\n{sample.to_csv(index=False)}\n\n"
        "User Request:\n" + user_prompt + "\n\n"
        "Please refine the above user request into a clear, step-by-step set of user-friendly transformation instructions (no formatting at all)"
        "that can be directly used by PandasAI to transform the entire dataset. "
        "Reference the metadata (or data sample) in your transformation instructions if needed, and the last instruction "
        "should be to display the entire resulting DataFrame. Return a JSON object with 1 key: 'refined_prompt'.\n"
        "IMPORTANT: The transformation instructions must be applied to the entire dataset. "
        "If any SQL queries are required, do not execute them directly. Instead, wrap your SQL query by calling "
        "the function `execute_sql_query('YOUR_SQL_QUERY')`."
    )
    return composite_prompt


class GoogleGeminiLLM(LLM):
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key

    def call(self, prompt: str, context=None, **kwargs) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        raw_response = response.text if response else "print('No response from Gemini')"
        # Remove markdown code block formatting, if any
        clean_response = re.sub(r"```(?:python)?\n(.*?)```", r"\1", raw_response, flags=re.DOTALL).strip()
        clean_response = remove_ansi_escape(clean_response)
        return clean_response

    def generate_code(self, prompt, context=None):
        return self.call(str(prompt), context)

    @property
    def type(self):
        return "Google Gemini"

    def refine_prompt(self, composite_prompt: str) -> dict:
        response_text = self.call(composite_prompt).strip()
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
        if response_text.lower().startswith("json"):
            response_text = response_text[len("json"):].strip()
        try:
            refined_output = json.loads(response_text)
            if isinstance(refined_output.get("refined_prompt", ""), list):
                refined_output["refined_prompt"] = "\n".join(refined_output["refined_prompt"])
        except Exception as e:
            refined_output = {
                "refined_prompt": "",
                "sample_results": f"Error parsing LLM response: {e}. Raw response: {response_text}"
            }
        return refined_output

gemini_llm = GoogleGeminiLLM(api_key=st.secrets["google"]["api_key"])

st.title("Excelerate")
st.write("Upload a CSV or Excel file, refine your transformation prompt (or use your original prompt), and preview/download the modified data.")

# File uploader for CSV or Excel files
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            # Read file bytes and detect encoding
            file_bytes = uploaded_file.read()
            encoding_info = chardet.detect(file_bytes)
            detected_encoding = encoding_info.get('encoding', 'utf-8')
            # Reset file pointer before reading CSV
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=detected_encoding)
        else:
            df = pd.read_excel(uploaded_file,engine='openpyxl')
        
        df.columns = df.columns.str.replace(r'\W+', '_', regex=True)
        df.columns = [remove_ansi_escape(col) for col in df.columns]
        
        st.subheader("Original Data")
        st.dataframe(df)
        
        metadata, sample = extract_metadata_and_sample(df, sample_size=5)
        st.session_state["metadata"] = metadata
        st.session_state["sample"] = sample
        
        st.subheader("Data Sample")
        st.dataframe(sample)
        
        user_prompt = st.text_area("What do you want to do with this table?",
                                   "e.g., remove the second column and filter rows where age > 40")
        
        
        if st.button("Refine Prompt", key="refine"):
            composite_prompt = create_composite_prompt(user_prompt, st.session_state["metadata"], st.session_state["sample"])
            refined_output = gemini_llm.refine_prompt(composite_prompt)
            refined_prompt = refined_output.get("refined_prompt", "")
            # Save the refined prompt in session state
            st.session_state["refined_prompt"] = refined_prompt
            st.subheader("Refined Prompt for PandasAI")
            # Provide an editable text area for the refined prompt
            st.text_area("Edit refined prompt if needed", value=refined_prompt, key="final_prompt")
        
            
            try:
                sample_sdf = SmartDataframe(st.session_state["sample"], config={"llm": gemini_llm})
                sample_response = sample_sdf.chat(refined_prompt)
                st.subheader("Sample Transformation Preview")
                if isinstance(sample_response, pd.DataFrame):
                    st.dataframe(sample_response)
                elif hasattr(sample_response, "value"):
                    st.dataframe(sample_response.value)
                else:
                    st.write(sample_response)
            except Exception as e:
                st.write("Error running transformation on sample data: ", e)
        
        
        if st.button("Process Data", key="process"):
            # Use the user-edited refined prompt if available; otherwise, if not refined, use the user prompt directly.
            final_prompt = st.session_state.get("final_prompt", st.session_state.get("refined_prompt", ""))
            if not final_prompt:
                final_prompt = user_prompt
            if not final_prompt:
                st.error("Please provide a prompt.")
            else:
                # Process the full original DataFrame
                sdf = SmartDataframe(df, config={"llm": gemini_llm})
                try:
                    final_response = sdf.chat(final_prompt)
                    if isinstance(final_response, pd.DataFrame):
                        modified_df = final_response
                    elif hasattr(final_response, "value"):
                        modified_df = final_response.value
                    else:
                        st.error("The response from PandasAI could not be converted to a DataFrame.")
                        modified_df = None
                    
                    if modified_df is not None:
                        st.subheader("Transformed Data")
                        st.dataframe(modified_df)
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = modified_df.to_csv(index=False).encode('utf-8')
                            st.download_button(label="Download CSV",
                                               data=csv,
                                               file_name="modified_data.csv",
                                               mime="text/csv")
                        with col2:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                modified_df.to_excel(writer, index=False, sheet_name='Sheet1')
                                writer.book.active.sheet_state = 'visible'
                            output.seek(0)
                            st.download_button(label="Download Excel",
                                               data=output.getvalue(),
                                               file_name="modified_data.xlsx",
                                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    
                    
                    clear_session_keys(["metadata", "sample", "refined_prompt", "final_prompt"])
                except Exception as e:
                    st.error("Error processing refined prompt: " + str(e))
    except Exception as e:
        st.error("Error loading file: " + str(e))
