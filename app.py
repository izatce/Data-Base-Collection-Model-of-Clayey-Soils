import os
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Clayey Soil Literature Search",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔎 Real Clayey Soil Literature Search")

st.caption(
    "Automatically searches different researchers and public sources "
    "for REAL reported laboratory data. No ANN prediction is used."
)

st.info(
    """
    **Research rule:** Only values explicitly reported in the original
    publication/report are accepted. Missing properties are recorded as
    **NR (Not Reported)**. The system must not estimate or fabricate
    laboratory results.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔬 Literature Search Settings")

    aashto_classes = st.multiselect(
        "AASHTO Soil Classes",
        [
            "A-6",
            "A-7-5",
            "A-7-6"
        ],
        default=[
            "A-6",
            "A-7-5",
            "A-7-6"
        ]
    )

    st.subheader("Properties to collect")

    properties = st.multiselect(
        "Select properties",
        [
            "Liquid Limit (LL)",
            "Plastic Limit (PL)",
            "Plasticity Index (PI)",
            "Clay %",
            "Silt %",
            "Sand %",
            "Gravel %",
            "Passing No. 200 %",
            "Specific Gravity (Gs)",
            "Natural Water Content",
            "OMC",
            "MDD",
            "Soaked CBR",
            "Unsoaked CBR",
            "UCS",
            "Cohesion",
            "Friction Angle"
        ],
        default=[
            "Liquid Limit (LL)",
            "Plastic Limit (PL)",
            "Plasticity Index (PI)",
            "Clay %",
            "Specific Gravity (Gs)",
            "Natural Water Content",
            "OMC",
            "MDD",
            "Soaked CBR",
            "Unsoaked CBR",
            "UCS",
            "Cohesion",
            "Friction Angle"
        ]
    )

    country = st.text_input(
        "Country / Region",
        placeholder="Pakistan, India, Nigeria, Ethiopia..."
    )

    location = st.text_input(
        "Specific location (optional)",
        placeholder="Jamshoro, Sindh..."
    )

    max_records = st.slider(
        "Maximum records per search",
        min_value=5,
        max_value=30,
        value=10
    )

    st.subheader("Search type")

    search_type = st.selectbox(
        "Literature search",
        [
            "General geotechnical properties",
            "CBR + Compaction",
            "UCS",
            "Direct Shear",
            "Index Properties",
            "All Properties"
        ]
    )

    search_button = st.button(
        "🔎 SEARCH REAL DATA",
        type="primary",
        use_container_width=True
    )


# ============================================================
# SEARCH PROMPT
# ============================================================

def build_search_prompt():

    class_text = ", ".join(aashto_classes)

    property_text = ", ".join(properties)

    country_text = (
        country
        if country
        else "No country restriction"
    )

    location_text = (
        location
        if location
        else "No specific location restriction"
    )

    return f"""

You are an expert geotechnical engineering literature-data
collection agent.

Your task is to SEARCH THE PUBLIC WEB and find REAL LABORATORY
DATA reported by different researchers.

TARGET AASHTO CLASSES:

{class_text}

TARGET PROPERTIES:

{property_text}

SEARCH TYPE:

{search_type}

COUNTRY:

{country_text}

LOCATION:

{location_text}


============================================================
IMPORTANT RESEARCH RULES
============================================================

1. Search MULTIPLE independent researchers.

2. Search original scientific papers, conference papers,
   theses, government reports, research reports and publicly
   available datasets.

3. Prefer the ORIGINAL SOURCE where the laboratory data were
   actually measured.

4. Do NOT rely only on review papers.

5. Do NOT use ANN, Random Forest, SVM, regression or other
   machine-learning predictions as laboratory measurements.

6. Do NOT estimate missing values.

7. Do NOT interpolate missing values.

8. Do NOT calculate an engineering property from another
   property unless the source itself explicitly reports that
   calculated value.

9. If a property is not reported, write:

   NR

10. Preserve the ORIGINAL reported value and UNIT.

11. Clearly distinguish:
    - Natural soil
    - Remoulded soil
    - Stabilized soil
    - Treated soil
    - Soaked CBR
    - Unsoaked CBR

12. For this database, natural/untreated soil should be
    preferred.

13. Preserve the published AASHTO classification.

14. Do NOT silently reclassify a soil.

15. Every record must have:

    Author
    Year
    Paper title
    Journal/report
    Country
    Location
    AASHTO classification
    USCS classification if available
    Source URL
    DOI if available
    Table/Figure/Section/Page provenance

16. If the numerical value cannot be verified from the
    source, return NR.

17. Avoid duplicate soil specimens.

18. One paper may contain multiple soil samples.
    Each actual sample should be a separate record.

19. The final database must contain REAL observations,
    NOT AI-generated data.

20. Search at least several different researchers before
    finishing the search.

21. For important values, prefer visiting the original
    source webpage/PDF when possible.

22. Maximum records:

{max_records}


============================================================
OUTPUT FORMAT
============================================================

Return VALID JSON only.

Use this structure:

{{
    "records": [

        {{
            "Record_ID": "",
            "Source_ID": "",

            "Authors": "",
            "Year": "",

            "Paper_Title": "",
            "Journal_or_Report": "",

            "Country": "",
            "Location": "",

            "Soil_Description": "",

            "AASHTO_Class": "",
            "USCS_Class": "",

            "Gravel_pct": "",
            "Sand_pct": "",
            "Silt_pct": "",
            "Clay_pct": "",

            "Passing_No200_pct": "",

            "LL_pct": "",
            "PL_pct": "",
            "PI_pct": "",

            "Gs": "",

            "Natural_Water_pct": "",

            "OMC_pct": "",
            "MDD": "",

            "CBR_Unsoaked_pct": "",
            "CBR_Soaked_pct": "",

            "UCS_kPa": "",

            "Cohesion_kPa": "",
            "Phi_deg": "",

            "Test_Condition": "",

            "Compaction_Standard": "",
            "CBR_Standard": "",

            "Data_Status": "Measured",

            "Source_URL": "",
            "DOI": "",

            "Provenance": ""
        }}

    ]
}}


============================================================
FINAL WARNING
============================================================

If the source does not explicitly report a value:

RETURN:

NR

Never guess.
Never fabricate.
Never use an ML prediction as a measured laboratory result.

"""


# ============================================================
# GROQ SEARCH FUNCTION
# ============================================================

def search_literature(prompt):

    client = Groq(
        api_key=GROQ_API_KEY,
        default_headers={
            "Groq-Model-Version": "latest"
        }
    )

    response = client.chat.completions.create(

        model="groq/compound",

        messages=[

            {
                "role": "system",
                "content": """
You are a strict scientific literature-data extraction
assistant specializing in geotechnical engineering.

Accuracy and source traceability are more important than
completeness.

Never invent laboratory measurements.
"""
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0,

        response_format={
            "type": "json_object"
        },

        compound_custom={
            "tools": {
                "enabled_tools": [
                    "web_search",
                    "visit_website"
                ]
            }
        }
    )

    return response


# ============================================================
# EXTRACT SEARCH SOURCES
# ============================================================

def extract_sources(response):

    sources = []

    try:

        message = response.choices[0].message

        executed_tools = getattr(
            message,
            "executed_tools",
            None
        )

        if executed_tools:

            for tool in executed_tools:

                if hasattr(tool, "search_results"):

                    results = tool.search_results

                    if results:

                        sources.append(
                            str(results)
                        )

    except Exception:
        pass

    return sources


# ============================================================
# SEARCH BUTTON
# ============================================================

if search_button:

    if not GROQ_API_KEY:

        st.error(
            "GROQ_API_KEY was not found."
        )

        st.stop()


    if not aashto_classes:

        st.warning(
            "Please select at least one AASHTO class."
        )

        st.stop()


    try:

        with st.spinner(
            "🔎 Searching researchers, papers and laboratory datasets..."
        ):

            prompt = build_search_prompt()

            response = search_literature(
                prompt
            )

            raw_response = (
                response
                .choices[0]
                .message
                .content
            )

            result = json.loads(
                raw_response
            )

            records = result.get(
                "records",
                []
            )

            result_df = pd.DataFrame(
                records
            )


            # Store results
            st.session_state[
                "literature_records"
            ] = result_df


            # Store sources
            st.session_state[
                "literature_sources"
            ] = extract_sources(
                response
            )


        st.success(
            f"Search completed. "
            f"{len(result_df)} real literature records found."
        )


    except Exception as e:

        error_text = str(e)

        if (
            "429" in error_text
            or "rate" in error_text.lower()
            or "quota" in error_text.lower()
        ):

            st.error(
                """
                Groq API rate limit or quota was reached.

                Please wait and try again with fewer records.
                """
            )

        else:

            st.error(
                f"Search failed:\n\n{error_text}"
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

result_df = st.session_state.get(
    "literature_records"
)


if (
    result_df is not None
    and not result_df.empty
):

    st.subheader(
        "📚 Real Literature Records Found"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        height=600
    )


    # --------------------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------------------

    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "⬇️ Download Search Results CSV",

        csv_data,

        "real_clayey_soil_literature.csv",

        "text/csv",

        use_container_width=True
    )


    # --------------------------------------------------------
    # SOURCE INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🔗 Search Sources"
    )

    sources = st.session_state.get(
        "literature_sources",
        []
    )

    if sources:

        for i, source in enumerate(
            sources,
            start=1
        ):

            with st.expander(
                f"Source search result {i}"
            ):

                st.text(
                    source
                )

    else:

        st.info(
            "Source details are retained in the Source_URL, DOI "
            "and Provenance columns."
        )


    # --------------------------------------------------------
    # DATA QUALITY CHECK
    # --------------------------------------------------------

    st.subheader(
        "✅ Data Quality Check"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Records",
        len(result_df)
    )

    if "Source_ID" in result_df.columns:

        col2.metric(
            "Sources",
            result_df[
                "Source_ID"
            ].nunique()
        )

    if "AASHTO_Class" in result_df.columns:

        col3.metric(
            "AASHTO groups",
            result_df[
                "AASHTO_Class"
            ].nunique()
        )

    if "Authors" in result_df.columns:

        col4.metric(
            "Researchers",
            result_df[
                "Authors"
            ].nunique()
        )


# ============================================================
# RESEARCH WORKFLOW
# ============================================================

st.subheader(
    "📊 Recommended Literature Data Collection Workflow"
)

st.markdown(
"""
### Batch 1
**A-6 soils**

### Batch 2
**A-7-5 soils**

### Batch 3
**A-7-6 soils**

### Batch 4
**CBR + OMC + MDD**

### Batch 5
**UCS**

### Batch 6
**Direct shear → cohesion + φ**

### Batch 7
**LL + PL + PI + gradation + Gs + natural water content**

The downloaded files can then be combined into your master
PhD literature database.
"""
)


# ============================================================
# IMPORTANT RESEARCH NOTE
# ============================================================

st.warning(
"""
### ⚠️ Important

This application is a **literature-data collection assistant**.

Before using a value in your PhD thesis/database:

1. Open the original paper/report.
2. Confirm the soil sample.
3. Confirm AASHTO classification.
4. Confirm the numerical value.
5. Confirm units.
6. Confirm test condition.
7. Confirm table/figure/page.
8. Keep the Source_ID.

**NR = Not Reported.**

Do not replace NR with an estimated value.
"""
)
