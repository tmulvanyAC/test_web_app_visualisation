import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Average Spectrum Viewer", layout="wide")

# Initialize session state for comparisons
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = []

@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    
    # Replace "0" or 0 with "Unknown" for subtype columns
    for col in ["HistologicalSubtype", "MolecularSubtype"]:
        if col in df.columns:
            df[col] = df[col].replace({0: "Unknown", 0.0: "Unknown", "0": "Unknown", "0.0": "Unknown"})
            
    return df

@st.cache_data
def apply_partial_auc_normalization(df, cols, ppm_vals):
    """Normalises spectra by dividing by the area under the curve between 1.5 and 4.0 ppm."""
    norm_df = df.copy()
    spectra = norm_df[cols]
    
    # Identify columns between 1.5 and 4.0 ppm
    valid_cols = [c for c, val in zip(cols, ppm_vals) if 1.5 <= val <= 4.0]
    
    # Sum only the valid region, replace 0 with alpha 10^-9 to avoid division by zero
    row_sums = spectra[valid_cols].sum(axis=1).replace(0, 1e-9)
    norm_df[cols] = spectra.div(row_sums, axis=0)
    
    return norm_df

#%% File Upload & Data Loading
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload spectra CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info("Please upload your spectrum spreadsheet (CSV) in the sidebar to begin.")
    st.stop()

# Define metadata and ppm columns
metadata_cols = [
    "ID", "FieldStrength", "PatientAge", "PatientGender", 
    "TumourLocation", "TumourType", "HistologicalSubtype", 
    "MolecularSubtype", "TumourGrade", "Site"
]
ppm_cols = [col for col in df.columns if col not in metadata_cols]
ppm_values = [float(p) for p in ppm_cols]

y_axis_label = "Normalized Intensity (Partial AUC)"

#%%Filtering Options
st.sidebar.header("Filtering Options")

tumour_types = df["TumourType"].dropna().unique()
selected_diagnosis = st.sidebar.selectbox("Diagnosis", sorted(tumour_types))

#Filter to base diagnosis first
base_df = df[df["TumourType"] == selected_diagnosis]

if base_df.empty:
    st.warning("No data found for the selected diagnosis.")
    st.stop()

# Apply Partial AUC Normalization (1.5-4.0 ppm)
base_df = apply_partial_auc_normalization(base_df, ppm_cols, ppm_values)

# Field Strength
field_strengths = base_df["FieldStrength"].dropna().unique()
selected_field_strengths = st.sidebar.multiselect("Field Strength", field_strengths, default=field_strengths)

# Location
locations = base_df["TumourLocation"].dropna().unique()
selected_locations = st.sidebar.multiselect("Location", locations, default=locations)

# Gender
genders = base_df["PatientGender"].dropna().unique()
selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)

# Age
min_age = float(base_df["PatientAge"].min(skipna=True))
max_age = float(base_df["PatientAge"].max(skipna=True))
if pd.isna(min_age) or pd.isna(max_age):
    selected_age_range = (0.0, 100.0)
    st.sidebar.warning("Age data missing for this diagnosis.")
elif min_age == max_age:
    selected_age_range = st.sidebar.slider("Age", 0.0, float(max_age + 10), (min_age, max_age))
else:
    selected_age_range = st.sidebar.slider("Age", min_age, max_age, (min_age, max_age))

# Subtype Selection
subtype_mapping = {
    "Histological": "HistologicalSubtype", 
    "Molecular": "MolecularSubtype"
}

valid_subtype_cats = []
for disp_name, col_name in subtype_mapping.items():
    if col_name in base_df.columns:
        uniques = base_df[col_name].dropna().unique()
        if not (len(uniques) == 0 or (len(uniques) == 1 and uniques[0] == "Unknown")):
            valid_subtype_cats.append(disp_name)

subtype_category = "None"
selected_subtypes = []

if valid_subtype_cats:
    subtype_category = st.sidebar.selectbox("Filter by Subtype", ["None"] + valid_subtype_cats)
    
    if subtype_category != "None":
        active_col = subtype_mapping[subtype_category]
        available_subtypes = base_df[active_col].dropna().unique()
        selected_subtypes = st.sidebar.multiselect(
            f"Select {subtype_category} to include:", 
            available_subtypes, 
            default=available_subtypes
        )


#%% Apply Filters
filtered_df = base_df.copy()
filtered_df = filtered_df[filtered_df["FieldStrength"].isin(selected_field_strengths)]
filtered_df = filtered_df[filtered_df["TumourLocation"].isin(selected_locations)]
filtered_df = filtered_df[filtered_df["PatientGender"].isin(selected_genders)]
filtered_df = filtered_df[
    (filtered_df["PatientAge"] >= selected_age_range[0]) & 
    (filtered_df["PatientAge"] <= selected_age_range[1])
]
if subtype_category != "None" and len(selected_subtypes) > 0:
    active_col = subtype_mapping[subtype_category]
    filtered_df = filtered_df[filtered_df[active_col].isin(selected_subtypes)]


#%% Define Functional Tabs
tab_search, tab_compare = st.tabs(["🔍 Search & Filter", "📊 Compare Spectra"])

#Search Tab
with tab_search:
    st.write(f"### {selected_diagnosis}")
    st.write(f"**Filtered Spectra count:** {len(filtered_df)} / {len(base_df)} total")

    if not filtered_df.empty:
        # Calculate Median and IQR
        median_spectrum = filtered_df[ppm_cols].median()
        q25 = filtered_df[ppm_cols].quantile(0.25)
        q75 = filtered_df[ppm_cols].quantile(0.75)
        
        # Build comparison name
        filter_desc = f"{selected_diagnosis} (n={len(filtered_df)})"
        if selected_age_range[0] != min_age or selected_age_range[1] != max_age:
            filter_desc += f" | Age {selected_age_range[0]:.0f}-{selected_age_range[1]:.0f}"
        if subtype_category != "None" and len(selected_subtypes) > 0:
             filter_desc += f" | {','.join([str(s) for s in selected_subtypes])}"
        if len(selected_field_strengths)==1:
            filter_desc += f" | {selected_field_strengths[0]}T"
        
        if st.button("➕ Add to Comparison"):
            st.session_state.comparisons.append({
                "name": filter_desc,
                "median": median_spectrum,
                "q25": q25,
                "q75": q75
            })
            st.success(f"Added '{filter_desc}' to Compare tab!")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ppm_values, y=q75, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=ppm_values, y=q25, mode='lines', line=dict(width=0), 
            fill='tonexty', fillcolor='rgba(0, 114, 178, 0.2)', name="IQR (25th - 75th)"
        ))
        fig.add_trace(go.Scatter(
            x=ppm_values, y=median_spectrum, mode='lines', 
            name="Median Spectrum", line=dict(color='rgba(0, 114, 178, 1)', width=2)
        ))
        
        fig.update_layout(
            title="Aggregated Pre-processed Spectrum (Median & IQR)",
            xaxis_title="Chemical Shift (ppm)", 
            yaxis_title=y_axis_label,
            xaxis=dict(autorange="reversed"), 
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches the current filter combination.")

with tab_compare:
    st.write("### Comparison View")
    
    if len(st.session_state.comparisons) == 0:
        st.info("No spectra added for comparison yet. Use the 'Add to Comparison' button in the Search tab.")
    else:
        if st.button("🗑️ Clear All Comparisons"):
            st.session_state.comparisons = []
            st.rerun()

        fig_comp = go.Figure()
        
        colors = [
            ('rgba(31, 119, 180, 1)', 'rgba(31, 119, 180, 0.2)'),  # Blue
            ('rgba(255, 127, 14, 1)', 'rgba(255, 127, 14, 0.2)'),   # Orange
            ('rgba(44, 160, 44, 1)', 'rgba(44, 160, 44, 0.2)'),     # Green
            ('rgba(214, 39, 40, 1)', 'rgba(214, 39, 40, 0.2)'),     # Red
            ('rgba(148, 103, 189, 1)', 'rgba(148, 103, 189, 0.2)')  # Purple
        ]

        for i, comp in enumerate(st.session_state.comparisons):
            line_color, fill_color = colors[i % len(colors)]
            
            # Confidence bounds
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["q75"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["q25"], mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=fill_color, showlegend=False, hoverinfo='skip'
            ))
            
            # Center line
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["median"], mode='lines', 
                name=comp["name"], line=dict(color=line_color, width=2)
            ))

        fig_comp.update_layout(
            title="Overlaid Spectra (Median & IQR)",
            xaxis_title="Chemical Shift (ppm)", 
            yaxis_title=y_axis_label,
            xaxis=dict(autorange="reversed"), 
            hovermode="x unified",
            legend=dict(
                traceorder="reversed",
                yanchor="top", y=1, xanchor="left", x=1.02
            )
        )
        st.plotly_chart(fig_comp, use_container_width=True)
