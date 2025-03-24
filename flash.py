import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.base import LLM
import google.generativeai as genai
import re

# ✅ Configure Google Gemini API Key
genai.configure(api_key="")

# ✅ Custom Google Gemini LLM
class GoogleGeminiLLM(LLM):
    def __init__(self, model_name="gemini-2.0-flash", api_key=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key

    def call(self, prompt: str, context=None, **kwargs) -> str:
        """Sends a plain-text prompt to Gemini and ensures it returns clean Python code."""
        model = genai.GenerativeModel(self.model_name)
        
        # Ensure prompt is a string and explicitly ask for Python code only
        prompt_text = (
            f"{prompt}\n\n"
            "Return only valid Python code with no explanations, no JSON, and no Markdown formatting. "
            "Ensure the output is directly executable without additional context."
        )
        
        response = model.generate_content(prompt_text)
        raw_code = response.text if response else "print('No response from Gemini')"
        print("\n🔹 Raw Gemini Output:\n", raw_code, "\n🔹 End of Output\n")

        # ✅ Strip Markdown formatting (```python ... ```)
        clean_code = re.sub(r"```(?:python)?(.*?)```", r"\1", raw_code, flags=re.DOTALL).strip()

        return clean_code

    def generate_code(self, prompt, context=None):
        """Ensure Gemini receives a plain text instruction and returns raw Python code."""
        return self.call(str(prompt), context)

    @property
    def type(self):
        return "Google Gemini"

# ✅ Initialize Gemini LLM
gemini_llm = GoogleGeminiLLM(api_key="AIzaSyCPM4iPRO1FrL1SduszTktoR_W1huGu1TA")

# ✅ Create DataFrame
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Salary": [50000, 60000, 70000]
})

# ✅ Wrap DataFrame with PandasAI
sdf = SmartDataframe(df, config={"llm": gemini_llm})

# ✅ Ask Gemini a question via PandasAI
try:
    response = sdf.chat("add a new column, age*salary and sort it bt largest to smallest")
    print(response)
except Exception as e:
    print("Error:", e)
