REAL CLAYEY SOIL LITERATURE SEARCH APP

This version automatically searches the public web using Gemini Google Search
grounding and extracts traceable laboratory data for A-6, A-7-5 and A-7-6 soils.

Setup:
1. Install requirements.txt
2. Create .streamlit/secrets.toml
3. Add:
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
4. Run:
   streamlit run app.py

Important:
- The model is instructed to use only explicitly reported numerical values.
- Missing values must be NR.
- It must not use ANN/ML predictions as laboratory measurements.
- Every record must contain Source_ID and Provenance.
- Verify the original paper/report before adding a record to the final PhD dataset.
