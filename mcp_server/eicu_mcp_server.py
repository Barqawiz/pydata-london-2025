"""
MCP Server for eICU Medical Data
Loads all patient files and serves complete patient data (including actual outcomes)
"""
import os
import sys
import pandas as pd
from pathlib import Path
from intelli.mcp import PandasMCPServerBuilder, PANDAS_AVAILABLE


def load_complete_patient_data():
    """Load and merge all patient data files including actual outcomes"""
    data_dir = Path("eicu_demo_data")

    files = {
        "patient": "patient.csv",
        "apache_result": "apachePatientResult.csv",
        "apache_vars": "apacheApsVar.csv",
        "labs": "lab.csv",
        "vitals": "vitalPeriodic.csv",
    }

    # Check files exist
    missing = []
    for name, filename in files.items():
        if not (data_dir / filename).exists():
            missing.append(filename)

    if missing:
        print(f"Missing files: {missing}")
        return None

    # Load core data
    patient_df = pd.read_csv(data_dir / files["patient"])
    apache_df = pd.read_csv(data_dir / files["apache_result"])
    apache_vars_df = pd.read_csv(data_dir / files["apache_vars"])

    # Merge core data
    merged_df = patient_df.merge(apache_df, on="patientunitstayid", how="left")
    merged_df = merged_df.merge(apache_vars_df, on="patientunitstayid", how="left")

    # Add lab counts per patient
    try:
        labs_df = pd.read_csv(data_dir / files["labs"])
        if not labs_df.empty:
            # Get lab counts and critical lab flags
            lab_counts = (
                labs_df.groupby("patientunitstayid")
                .size()
                .reset_index(name="lab_count")
            )
            merged_df = merged_df.merge(lab_counts, on="patientunitstayid", how="left")

            # Add critical lab flags
            critical_labs = ["wbc", "creatinine", "lactate", "bilirubin", "glucose"]
            for lab in critical_labs:
                lab_patients = labs_df[
                    labs_df["labname"].str.lower().str.contains(lab, na=False)
                ]["patientunitstayid"].unique()
                merged_df[f"has_{lab}"] = merged_df["patientunitstayid"].isin(
                    lab_patients
                )
    except:
        merged_df["lab_count"] = 0

    # Add vitals stats per patient
    try:
        vitals_df = pd.read_csv(data_dir / files["vitals"])
        if not vitals_df.empty:
            vitals_stats = (
                vitals_df.groupby("patientunitstayid")
                .agg(
                    {
                        "heartrate": ["mean", "max"],
                        "systemicsystolic": ["mean", "max"],
                        "temperature": ["mean", "max"],
                    }
                )
                .round(2)
            )

            vitals_stats.columns = [
                "_".join(col).strip() for col in vitals_stats.columns
            ]
            vitals_stats = vitals_stats.reset_index()
            merged_df = merged_df.merge(
                vitals_stats, on="patientunitstayid", how="left"
            )
    except:
        pass

    # Add convenient mortality flag (keep actual outcomes for comparison)
    merged_df["expired"] = merged_df["actualicumortality"].str.contains(
        "EXPIRED", case=False, na=False
    )

    print(
        f"Loaded complete dataset: {len(merged_df)} patients with {len(merged_df.columns)} features"
    )
    print(
        "Dataset includes actual outcomes - preprocessor will clean data for prediction"
    )

    return merged_df


def main():
    if not PANDAS_AVAILABLE:
        print("Need pandas: pip install pandas")
        sys.exit(1)

    # Load complete data (including actual outcomes)
    complete_data = load_complete_patient_data()
    if complete_data is None:
        sys.exit(1)

    print("Available columns:")
    for col in complete_data.columns:
        print(f"  {col}")

    # Save complete data for MCP server
    temp_file = "complete_patient_data.csv"
    complete_data.to_csv(temp_file, index=False)

    # Setup MCP server with complete data
    server = PandasMCPServerBuilder(
        server_name="CompleteMedicalDataServer",
        csv_file_path=temp_file,
        stateless_http=True,
    )

    if server.df is None:
        print("Failed to create server")
        sys.exit(1)

    print(f"\nMCP Server ready with {len(server.df)} patients")
    print("Server URL: http://localhost:8000/mcp")
    print("Operations: filter_rows (by patient ID), get_schema, get_head")
    print("------")
    print("\nExample client usage:")
    print("  model_params = {")
    print("    'url': 'http://localhost:8000/mcp',")
    print("    'tool': 'filter_rows',")
    print("    'arg_column': 'patientunitstayid',")
    print("    'arg_operator': '==',")
    print("    'arg_value': 141296")
    print("  }")

    server.run(
        transport="streamable-http", mount_path="/mcp", host="0.0.0.0", port=8000
    )


if __name__ == "__main__":
    main()
