
import os, json, re
from pathlib import Path
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Real Clayey Soil Literature Search",
    page_icon="🔎",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "real_clayey_soil_literature_database.xlsx"

# -----------------------------
# Existing local database
# -----------------------------
@st.cache_data
def load_local_data():
    if DATA_FILE.exists():
        return pd.read_excel(DATA_FILE, sheet_name="MASTER_DATA")
    return pd.DataFrame()

@st.cache_data
def load_references():
    if DATA_FILE.exists():
        return pd.read_excel(DATA_FILE, sheet_name="REFERENCES")
    return pd.DataFrame()

local_df = load_local_data()
local_refs = load_references()

# -----------------------------
# Gemini client
# -----------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

st.title("🔎 Real Clayey Soil Literature Search")
st.caption(
    "Automatically searches the public web for published laboratory data "
    "for A-6, A-7-5 and A-7-6 soils. No ANN prediction is used."
)

st.warning(
    "Important: the search assistant is instructed to extract only values that "
    "are explicitly reported in the source. If a property cannot be verified, "
    "it must be returned as NR (Not Reported)."
)

# -----------------------------
# Search controls
# -----------------------------
with st.sidebar:
    st.header("Search criteria")

    aashto_classes = st.multiselect(
        "AASHTO classes",
        ["A-6", "A-7-5", "A-7-6"],
        default=["A-6", "A-7-5", "A-7-6"]
    )

    properties = st.multiselect(
        "Required / desired properties",
        [
            "LL", "PL", "PI", "Clay %", "Silt %", "Sand %",
            "Gravel %", "Passing No.200", "Gs",
            "Natural Water Content", "OMC", "MDD",
            "CBR soaked", "CBR unsoaked", "UCS",
            "Cohesion", "Phi"
        ],
        default=[
            "LL", "PL", "PI", "Clay %",
            "Gs", "Natural Water Content",
            "OMC", "MDD", "CBR soaked",
            "CBR unsoaked", "UCS", "Cohesion", "Phi"
        ]
    )

    max_sources = st.slider(
        "Maximum source records to request",
        min_value=5, max_value=50, value=20
    )

    country_hint = st.text_input(
        "Country/location filter (optional)",
        placeholder="Pakistan, India, Nigeria, Ethiopia..."
    )

    include_reports = st.checkbox(
        "Include theses/reports/open datasets",
        value=True
    )

    search_button = st.button(
        "🔎 Search real literature data",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Search prompt
# -----------------------------
def build_prompt():
    class_text = ", ".join(aashto_classes)
    prop_text = ", ".join(properties)
    report_rule = (
        "Include peer-reviewed papers, government reports, theses, and openly "
        "available research datasets when they contain traceable laboratory observations."
        if include_reports else
        "Prefer peer-reviewed journal/conference papers and exclude theses/reports unless unavoidable."
    )

    return f"""
You are a rigorous geotechnical literature-data extraction agent.

TASK:
Search the public web and identify REAL, traceable laboratory observations for
clayey/fine-grained soils classified as:
{class_text}

The requested engineering/index properties are:
{prop_text}

{report_rule}

Country/location preference:
{country_hint if country_hint else "No country restriction."}

CRITICAL DATA RULES:
1. Search multiple independent sources/researchers. Do not rely on one paper.
2. Prefer the original paper/report/dataset over a secondary paper that merely
   cites another study.
3. Extract ONLY numerical values explicitly visible/reported by the source.
4. NEVER estimate, interpolate, calculate, infer, or invent a missing value.
5. If a requested property is not reported for a record, enter "NR".
6. Preserve the source's original value and unit.
7. Do not treat ANN/ML predictions as laboratory measurements.
8. Separate natural/untreated soil from stabilized soil. If a paper reports
   both, use the untreated/natural row when the objective is natural clayey soil.
9. Preserve the published AASHTO class. Do not silently reclassify it.
10. For every numerical record provide the exact source, DOI/URL if available,
    and the table/figure/section/page where the value was found.
11. Do not duplicate the same soil record merely because several papers cite it.
12. If a source cannot be opened or the value cannot be verified, do not use
    the value.

OUTPUT:
Return JSON only, with this structure:

{{
  "records": [
    {{
      "Record_ID": "...",
      "Source_ID": "...",
      "Authors": "...",
      "Year": 2020,
      "Paper_Title": "...",
      "Journal_or_Report": "...",
      "Country": "...",
      "Location": "...",
      "Soil_Description": "...",
      "AASHTO_Class": "...",
      "USCS_Class": "...",
      "Gravel_pct": "...",
      "Sand_pct": "...",
      "Silt_pct": "...",
      "Clay_pct": "...",
      "Passing_No200_pct": "...",
      "LL_pct": "...",
      "PL_pct": "...",
      "PI_pct": "...",
      "Gs": "...",
      "Natural_Water_pct": "...",
      "OMC_pct": "...",
      "MDD": "...",
      "CBR_Unsoaked_pct": "...",
      "CBR_Soaked_pct": "...",
      "UCS_kPa": "...",
      "Cohesion_kPa": "...",
      "Phi_deg": "...",
      "Test_Condition": "...",
      "Compaction_Standard": "...",
      "CBR_Standard": "...",
      "Data_Status": "Measured",
      "Source_URL": "...",
      "DOI": "...",
      "Provenance": "Table/Figure/Section/Page"
    }}
  ]
}}

Return at most {max_sources} records.
"""

# -----------------------------
# Execute grounded web search
# -----------------------------
if search_button:
    if not API_KEY:
        st.error(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets "
            "or the environment."
        )
        st.stop()

    try:
        client = genai.Client(api_key=API_KEY)

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ]
        )

        with st.spinner(
            "Searching multiple web sources and extracting only reported laboratory values..."
        ):
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=build_prompt(),
                config=config
            )

        raw = response.text
        result = json.loads(raw)
        records = result.get("records", [])

        st.session_state["search_records"] = records
        st.session_state["search_raw"] = raw

        st.success(
            f"Search completed. {len(records)} literature records were returned."
        )

    except Exception as e:
        st.error(f"Search/extraction failed: {e}")
        st.info(
            "If your Gemini account does not expose Gemini 2.5 Flash-Lite, change the "
            "model name in app.py to another model that supports Google Search grounding."
        )

# -----------------------------
# Display results
# -----------------------------
records = st.session_state.get("search_records", [])

if records:
    search_df = pd.DataFrame(records)

    st.subheader("🔬 Newly searched literature records")
    st.dataframe(search_df, use_container_width=True, height=550)

    st.download_button(
        "⬇️ Download searched records as CSV",
        search_df.to_csv(index=False).encode("utf-8"),
        "new_real_clayey_soil_literature_records.csv",
        "text/csv"
    )

    # Save a combined research database
    if st.button("➕ Add verified search records to database"):
        try:
            existing = local_df.copy()

            # Keep only matching master columns where possible.
            master_cols = [
                "Record_ID", "Source_ID", "Author_or_Organization", "Year",
                "Country", "Location", "AASHTO_Class", "USCS_Class",
                "Gravel_pct", "Sand_pct", "Silt_pct", "Clay_pct",
                "Passing_No200_pct", "LL_pct", "PL_pct", "PI_pct", "Gs",
                "Natural_Water_pct", "OMC_pct", "MDD",
                "CBR_Unsoaked_pct", "CBR_Soaked_pct", "UCS_kPa",
                "Cohesion_kPa", "Phi_deg", "Provenance"
            ]

            add = pd.DataFrame()
            for col in master_cols:
                source_col = col
                if col == "Author_or_Organization":
                    source_col = "Authors"
                if source_col in search_df:
                    add[col] = search_df[source_col]
                else:
                    add[col] = "NR"

            combined = pd.concat([existing, add], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["Record_ID", "Source_ID"],
                keep="first"
            )

            st.session_state["combined_df"] = combined
            st.success(
                f"Combined database now contains {len(combined)} records. "
                "Download it below and review source provenance before thesis use."
            )
        except Exception as e:
            st.error(f"Could not combine records: {e}")

combined_df = st.session_state.get("combined_df")

if combined_df is not None:
    st.subheader("📚 Combined literature database")
    st.dataframe(combined_df, use_container_width=True, height=450)

    st.download_button(
        "⬇️ Download combined database",
        combined_df.to_csv(index=False).encode("utf-8"),
        "real_clayey_soil_combined_literature_database.csv",
        "text/csv"
    )

# -----------------------------
# Existing database
# -----------------------------
with st.expander("View current local database"):
    if not local_df.empty:
        st.dataframe(local_df, use_container_width=True, height=400)
    else:
        st.info("No local database found.")

st.subheader("⚠️ Research verification protocol")
st.markdown("""
**The app searches first, but the researcher must verify the extracted values.**

For a value to be accepted into the final PhD dataset:

1. Open the original paper/report.
2. Confirm the soil is A-6, A-7-5 or A-7-6.
3. Confirm the numerical value in the cited table/figure/section.
4. Confirm whether the specimen is natural, remoulded, soaked, unsoaked or stabilized.
5. Confirm the test standard and compaction condition where available.
6. Keep the Source_ID and Provenance with the observation.
7. Use **NR** when the source does not report a property.

This prevents the AI from turning literature interpretation into fabricated laboratory data.
""")
