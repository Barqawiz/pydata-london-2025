"""
MCP Server for eICU Medical Data using Polars
Loads all patient files and serves complete patient data (including actual outcomes)
"""
import os
import sys
import polars as pl
from pathlib import Path
from intelli.mcp import PolarsMCPServerBuilder, POLARS_AVAILABLE


def load_complete_patient_data():
    """Load and merge all patient data files including actual outcomes using Polars"""
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
    patient_df = pl.read_csv(data_dir / files["patient"])
    apache_df = pl.read_csv(data_dir / files["apache_result"])
    apache_vars_df = pl.read_csv(data_dir / files["apache_vars"])

    # Merge core data using Polars joins
    merged_df = patient_df.join(apache_df, on="patientunitstayid", how="left")
    merged_df = merged_df.join(apache_vars_df, on="patientunitstayid", how="left")

    # Add lab counts per patient
    try:
        labs_df = pl.read_csv(data_dir / files["labs"])
        if labs_df.height > 0:
            # Get lab counts using Polars group_by
            lab_counts = (
                labs_df.group_by("patientunitstayid")
                .len()
                .rename({"len": "lab_count"})
            )
            merged_df = merged_df.join(lab_counts, on="patientunitstayid", how="left")

            # Add critical lab flags
            critical_labs = ["wbc", "creatinine", "lactate", "bilirubin", "glucose"]
            for lab in critical_labs:
                lab_patients = labs_df.filter(
                    pl.col("labname").str.to_lowercase().str.contains(lab)
                )["patientunitstayid"].unique()
                
                merged_df = merged_df.with_columns(
                    pl.col("patientunitstayid").is_in(lab_patients).alias(f"has_{lab}")
                )
    except Exception as e:
        print(f"Error processing labs: {e}")
        merged_df = merged_df.with_columns(pl.lit(0).alias("lab_count"))

    # Add vitals stats per patient
    try:
        vitals_df = pl.read_csv(data_dir / files["vitals"])
        if vitals_df.height > 0:
            vitals_stats = (
                vitals_df.group_by("patientunitstayid")
                .agg([
                    pl.col("heartrate").mean().round(2).alias("heartrate_mean"),
                    pl.col("heartrate").max().alias("heartrate_max"),
                    pl.col("systemicsystolic").mean().round(2).alias("systemicsystolic_mean"),
                    pl.col("systemicsystolic").max().alias("systemicsystolic_max"),
                    pl.col("temperature").mean().round(2).alias("temperature_mean"),
                    pl.col("temperature").max().alias("temperature_max"),
                ])
            )
            merged_df = merged_df.join(vitals_stats, on="patientunitstayid", how="left")
    except Exception as e:
        print(f"Error processing vitals: {e}")

    # Add convenient mortality flag (keep actual outcomes for comparison)
    merged_df = merged_df.with_columns(
        pl.col("actualicumortality")
        .str.to_lowercase()
        .str.contains("expired")
        .fill_null(False)
        .alias("expired")
    )

    print(
        f"Loaded complete dataset: {merged_df.height} patients with {merged_df.width} features"
    )
    print(
        "Dataset includes actual outcomes - preprocessor will clean data for prediction"
    )

    return merged_df


def main():
    if not POLARS_AVAILABLE:
        print("Need polars: pip install polars")
        sys.exit(1)

    # Load complete data (including actual outcomes)
    complete_data = load_complete_patient_data()
    if complete_data is None:
        sys.exit(1)

    print("Available columns:")
    for col in complete_data.columns:
        print(f"  {col}")

    # Save complete data for MCP server
    temp_file = "complete_patient_data_polars.csv"
    complete_data.write_csv(temp_file)

    # Setup MCP server with complete data
    server = PolarsMCPServerBuilder(
        server_name="CompleteMedicalDataServerPolars",
        csv_file_path=temp_file,
        stateless_http=True,
    )

    if server.df is None:
        print("Failed to create server")
        sys.exit(1)

    print(f"\nMCP Server ready with {server.df.height} patients")
    print("Server URL: http://localhost:8001/mcp")
    print("Operations: filter_rows (by patient ID), get_schema, get_head")
    print("------")
    print("\nExample client usage:")
    print("  model_params = {")
    print("    'url': 'http://localhost:8001/mcp',")
    print("    'tool': 'filter_rows',")
    print("    'arg_column': 'patientunitstayid',")
    print("    'arg_operator': '==',")
    print("    'arg_value': 141296")
    print("  }")

    server.run(
        transport="streamable-http", mount_path="/mcp", host="0.0.0.0", port=8001
    )


if __name__ == "__main__":
    main() 