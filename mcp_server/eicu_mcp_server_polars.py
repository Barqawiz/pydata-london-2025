"""
MCP Server for eICU Medical Data using Polars - FIXED VERSION
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

    # Remove duplicates from Apache data to prevent duplicate rows after merge
    apache_df = apache_df.unique(subset=["patientunitstayid"])
    apache_vars_df = apache_vars_df.unique(subset=["patientunitstayid"])

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
                )["patientunitstayid"].unique().to_list()  # Convert to list explicitly
                
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
            # Check which numeric columns actually exist
            available_numeric_cols = []
            for col in ["heartrate", "systemicsystolic", "temperature"]:
                if col in vitals_df.columns:
                    # Check if column is numeric or can be converted
                    try:
                        if vitals_df[col].dtype.is_numeric():
                            available_numeric_cols.append(col)
                        else:
                            # Try to convert to numeric
                            vitals_df = vitals_df.with_columns(
                                pl.col(col).cast(pl.Float64, strict=False).alias(col)
                            )
                            available_numeric_cols.append(col)
                    except:
                        print(f"Warning: Could not process vitals column {col}")
            
            if available_numeric_cols:
                # Build aggregation expressions dynamically
                agg_exprs = []
                for col in available_numeric_cols:
                    agg_exprs.extend([
                        pl.col(col).mean().round(2).alias(f"{col}_mean"),
                        pl.col(col).max().alias(f"{col}_max"),
                    ])
                
                vitals_stats = vitals_df.group_by("patientunitstayid").agg(agg_exprs)
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


class PolarsMCPServerBuilder(PolarsMCPServerBuilder):
    """Fixed version of PolarsMCPServerBuilder with better type handling"""
    
    def _filter_df_rows(self, column: str, operator: str, value) -> str:
        """Fixed version with better type conversion and debugging"""
        if self.df is None: 
            return "Error: DataFrame not loaded."
        if column not in self.df.columns:
            return f"Error: Column '{column}' not found."

        print(f"DEBUG: Filtering column '{column}' with operator '{operator}' and value '{value}' (type: {type(value)})")
        
        # Convert value to correct type for comparison
        try:
            col_dtype = self.df[column].dtype
            print(f"DEBUG: Column '{column}' has dtype: {col_dtype}")
            
            if operator == 'in':
                if not isinstance(value, list):
                    return "Error: For 'in' operator, value must be a list."
            elif col_dtype.is_integer():
                # Ensure we convert to the right integer type
                if isinstance(value, str):
                    value = int(value)
                elif isinstance(value, float):
                    value = int(value)
                print(f"DEBUG: Converted value to int: {value}")
            elif col_dtype.is_float():
                value = float(value)
                print(f"DEBUG: Converted value to float: {value}")
            elif col_dtype == pl.Boolean:
                if isinstance(value, str):
                    value = str(value).lower() in ['true', '1', 'yes']
                elif isinstance(value, bool):
                    pass  # already boolean
                else:
                    value = bool(value)
                print(f"DEBUG: Converted value to bool: {value}")
            elif col_dtype == pl.Utf8 and not isinstance(value, str):
                value = str(value)
                print(f"DEBUG: Converted value to str: {value}")

        except Exception as e:
            error_msg = f"Error converting value for filtering: {str(e)}. Column '{column}' type is {col_dtype}."
            print(f"DEBUG: {error_msg}")
            return error_msg

        # Apply the filter
        col_expr = pl.col(column)
        
        try:
            if operator == '==':
                condition = col_expr == value
            elif operator == '!=':
                condition = col_expr != value
            elif operator == '>':
                condition = col_expr > value
            elif operator == '<':
                condition = col_expr < value
            elif operator == '>=':
                condition = col_expr >= value
            elif operator == '<=':
                condition = col_expr <= value
            elif operator == 'contains':
                if not isinstance(value, str):
                    return "Error: 'contains' operator requires a string value."
                if self.df[column].dtype != pl.Utf8:
                     condition = col_expr.cast(pl.Utf8).str.contains(value, literal=False)
                else:
                     condition = col_expr.str.contains(value, literal=False)
            elif operator == 'in':
                if not isinstance(value, list):
                     return "Error: 'in' operator requires a list value."
                condition = col_expr.is_in(value)
            else:
                return f"Error: Unsupported operator '{operator}'. Supported operators are '==', '!=', '>', '<', '>=', '<=', 'contains', 'in'."
            
            filtered_df = self.df.filter(condition)
            print(f"DEBUG: Filter produced {filtered_df.height} rows")
            
            result_json = self._df_to_json(filtered_df)
            print(f"DEBUG: JSON result length: {len(result_json)}")
            
            return result_json
            
        except Exception as e:
            error_msg = f"Error applying filter: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return error_msg


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
    print("Server URL: http://localhost:8000/mcp")
    print("Operations: filter_rows (by patient ID), get_schema, get_head")
    print("------")
    print("\nExample client usage:")
    print("  model_params = {")
    print("    'url': 'http://localhost:8000/mcp',")
    print("    'tool': 'filter_rows',")
    print("    'arg_column': 'patientunitstayid',")
    print("    'arg_operator': '==',")
    print("    'arg_value': 2834225")
    print("  }")

    server.run(
        transport="streamable-http", mount_path="/mcp", host="0.0.0.0", port=8000
    )


if __name__ == "__main__":
    main() 