import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Imaging Repository Viewer", layout="wide")

# Page Title with description info tooltip
st.title(
    "Short-Echo Single Voxel Spectroscopy",
    help="SVS acquired at short echo times 30-41ms at diagnosis prior to surgery or commencement of treatment.",
)

# Metabolite Group Definitions
METABOLITE_GROUPS = {
    "TCho": ["PCh", "GPC"],
    "TCr": ["Cr", "PCr"],
    "TNAA": ["NAA", "NAAG"],
    "Glx": ["Glu", "Gln"],
    "TLM": [
        "Lip09", "Lip13a", "Lip13b", "Lip20",
        "MM09", "MM12", "MM14", "MM17", "MM20", "MM38"
    ],
    "Lac": ["Lac"],
    "m-Ins": ["Ins"],
    "Tau": ["Tau"],
    "Cit": ["Cit"],
    "Gly": ["Gly"],
    "GABA": ["GABA"],
    "PEth": ["PEth"],
    "Scyllo": ["Scyllo"],
}

METAB_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999"
]

if "comparisons" not in st.session_state:
    st.session_state.comparisons = []

@st.cache_data
def load_data(source):
    archive = np.load(source)
    spectra = archive["spectra"]
    ppm = archive["ppm"].astype(float)
    metabolites = [str(m) for m in archive["metabolites"]]
    meta_df = pd.read_json(str(archive["metadata_json"]), orient="split")

    for col in ["HistologicalSubtype", "MolecularSubtype", "TumourLocation"]:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].replace(
                {0: "Unknown", 0.0: "Unknown", "0": "Unknown", "0.0": "Unknown"}
            )

    return meta_df, spectra, ppm, metabolites


#%% File Upload & Data Loading/Mapping
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload data file (.npz)", type=["npz"])

if uploaded_file is not None:
    df, spectra, ppm_values, metabolite_names = load_data(uploaded_file)
else:
    st.info("Please upload your data archive (.npz) in the sidebar to begin.")
    st.stop()

# Pre-map available metabolites to their indices along axis 1
metab_to_idx = {name: i for i, name in enumerate(metabolite_names)}

# Locate the 'Fit' trace index
fit_idx = metab_to_idx.get("Fit", None)

# Map group display names to component indices (excluding 'Fit')
available_groups = {
    group: [metab_to_idx[m] for m in targets if m in metab_to_idx and m != "Fit"]
    for group, targets in METABOLITE_GROUPS.items()
}
active_group_options = [g for g, idxs in available_groups.items() if len(idxs) > 0]

y_axis_label = "Median Intensity (a.u.)"

#%% Filtering Options
st.sidebar.header("Filtering Options")

tumour_types = df["TumourType"].dropna().unique()
selected_diagnosis = st.sidebar.selectbox("Diagnosis", sorted(tumour_types))

base_mask = (df["TumourType"] == selected_diagnosis).to_numpy()
if not np.any(base_mask):
    st.warning("No data found for the selected diagnosis.")
    st.stop()

base_df = df[base_mask]

# Field Strength (Single selection: 1.5T or 3T)
field_strengths = sorted(base_df["FieldStrength"].dropna().unique())
selected_field_strength = st.sidebar.radio(
    "Field Strength (T)",
    options=field_strengths,
    format_func=lambda x: f"{x}T",
    horizontal=True,
)

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
    "Molecular": "MolecularSubtype",
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
            default=available_subtypes,
        )

#%% Apply Filters
filter_mask = (
    base_mask
    & (df["FieldStrength"] == selected_field_strength)
    & df["TumourLocation"].isin(selected_locations)
    & df["PatientGender"].isin(selected_genders)
    & (df["PatientAge"] >= selected_age_range[0])
    & (df["PatientAge"] <= selected_age_range[1])
)

if subtype_category != "None" and len(selected_subtypes) > 0:
    active_col = subtype_mapping[subtype_category]
    filter_mask &= df[active_col].isin(selected_subtypes)

filtered_df = df[filter_mask]
filtered_indices = np.where(filter_mask)[0]

#%% Define Functional Tabs
tab_search, tab_compare = st.tabs(["🔍 Search & Filter", "📊 Compare Spectra"])
#Search Tab
with tab_search:
    st.write(f"### {selected_diagnosis}")
    st.write(f"**Filtered Spectra count:** {len(filtered_df)} / {len(base_df)} total")

    if len(filtered_indices) > 0:
        cohort_tensor = spectra[filtered_indices]

        # Use pre-computed 'Fit' row directly if present; fallback to sum of non-fit basis rows
        if fit_idx is not None:
            fit_per_patient = cohort_tensor[:, fit_idx, :]
        else:
            basis_indices = [i for name, i in metab_to_idx.items() if name.lower() != "fit"]
            fit_per_patient = cohort_tensor[:, basis_indices, :].sum(axis=1)

        median_spectrum = np.median(fit_per_patient, axis=0)
        q25 = np.percentile(fit_per_patient, 25, axis=0)
        q75 = np.percentile(fit_per_patient, 75, axis=0)

        # Precompute cohort median for each active metabolite group
        cohort_metabolite_medians = {}
        for group_label, group_indices in available_groups.items():
            if len(group_indices) > 0:
                group_per_patient = cohort_tensor[:, group_indices, :].sum(axis=1)
                cohort_metabolite_medians[group_label] = np.median(group_per_patient, axis=0)

        # Metabolite contribution controls
        st.write("#### Metabolite Contributions")
        selected_groups = st.pills(
            "Toggle metabolites to display:",
            options=active_group_options,
            selection_mode="multi",
            default=[],
        )

        # Comparison label construction
        filter_desc = f"{selected_diagnosis} (n={len(filtered_df)}) | {selected_field_strength}T"
        if selected_age_range[0] != min_age or selected_age_range[1] != max_age:
            filter_desc += f" | Age {selected_age_range[0]:.0f}-{selected_age_range[1]:.0f}"
        if subtype_category != "None" and len(selected_subtypes) > 0:
            filter_desc += f" | {','.join(str(s) for s in selected_subtypes)}"

        if st.button("➕ Add to Comparison"):
            st.session_state.comparisons.append({
                "name": filter_desc,
                "median": median_spectrum,
                "q25": q25,
                "q75": q75,
                "metabolite_medians": cohort_metabolite_medians,
            })
            st.success(f"Added '{filter_desc}' to Compare tab!")

        # Main Plot
        fig = go.Figure()

        #Full Median + IQR
        fig.add_trace(go.Scatter(
            x=ppm_values, y=q25, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=ppm_values, y=q75, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0, 114, 178, 0.15)", name="Total Fit IQR"
        ))
        fig.add_trace(go.Scatter(
            x=ppm_values, y=median_spectrum, mode="lines",
            name="Total Median Fit",
            line=dict(color="rgba(0, 114, 178, 1)", width=2.5)
        ))

        #Selected Metabolite
        for i, group_label in enumerate(selected_groups):
            if group_label in cohort_metabolite_medians:
                group_median = cohort_metabolite_medians[group_label]
                color = METAB_COLORS[i % len(METAB_COLORS)]
                fig.add_trace(go.Scatter(
                    x=ppm_values,
                    y=group_median,
                    mode="lines",
                    name=group_label,
                    line=dict(color=color, width=1.75, dash="solid")
                ))

        fig.update_layout(
            title="Aggregated Spectrum & Metabolite Breakdown",
            xaxis_title="Chemical Shift (ppm)",
            yaxis_title=y_axis_label,
            xaxis=dict(autorange="reversed"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches the current filter combination.")

with tab_compare:
    st.write("### Comparison View")

    if len(st.session_state.comparisons) == 0:
        st.info("No spectra added for comparison yet. Use the 'Add to Comparison' button in the Search tab.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            compare_metabolite = st.pills(
                "Overlay Metabolite Contribution across Cohorts (limit 1):",
                options=["None"] + active_group_options,
                selection_mode="single",
                default="None"
            )
        with c2:
            st.write("")
            if st.button("🗑️ Clear All Comparisons"):
                st.session_state.comparisons = []
                st.rerun()

        fig_comp = go.Figure()

        colors = [
            ("rgba(31, 119, 180, 1)", "rgba(31, 119, 180, 0.2)"),
            ("rgba(255, 127, 14, 1)", "rgba(255, 127, 14, 0.2)"),
            ("rgba(44, 160, 44, 1)", "rgba(44, 160, 44, 0.2)"),
            ("rgba(214, 39, 40, 1)", "rgba(214, 39, 40, 0.2)"),
            ("rgba(148, 103, 189, 1)", "rgba(148, 103, 189, 0.2)"),
            ("rgba(140, 86, 75, 1)", "rgba(140, 86, 75, 0.2)"),
            ("rgba(227, 119, 194, 1)", "rgba(227, 119, 194, 0.2)"),
            ("rgba(188, 189, 34, 1)", "rgba(188, 189, 34, 0.2)"),
        ]

        for i, comp in enumerate(st.session_state.comparisons):
            line_color, fill_color = colors[i % len(colors)]

            # IQR bounds
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["q75"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["q25"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor=fill_color, showlegend=False, hoverinfo="skip"
            ))

            # Full Spectrum Line
            fig_comp.add_trace(go.Scatter(
                x=ppm_values, y=comp["median"], mode="lines",
                name=comp["name"], line=dict(color=line_color, width=2)
            ))

            # Optional metabolite overlay sharing the same line colour
            if compare_metabolite and compare_metabolite != "None":
                metab_dict = comp.get("metabolite_medians", {})
                if compare_metabolite in metab_dict:
                    fig_comp.add_trace(go.Scatter(
                        x=ppm_values,
                        y=metab_dict[compare_metabolite],
                        mode="lines",
                        name=f"{comp['name']} · {compare_metabolite}",
                        line=dict(color=line_color, width=1.2, dash="dot"),
                    ))

        fig_comp.update_layout(
            title="Spectral Comparison Between Selected Groups",
            xaxis_title="Chemical Shift (ppm)",
            yaxis_title=y_axis_label,
            xaxis=dict(autorange="reversed"),
            hovermode="x unified",
            legend=dict(
                traceorder="normal",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
        )
        st.plotly_chart(fig_comp, use_container_width=True)