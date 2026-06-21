import json
import polars as pl

def parse_data(input_path: str, output_path: str) -> None:
    """
    Parse JSONL file and save as Parquet with specific transformations.

    Steps:
    1. Read JSONL line by line using json module.
    2. Flatten profile and redrob_signals dictionaries.
    3. Convert skill_assessment_scores dict to JSON string.
    4. Split expected_salary_range_inr_lpa into min and max.
    5. Leave lists of structs (skills, career_history) as-is.
    6. Save as Parquet.
    """
    parsed_records = []

    with open(input_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())

            # Flatten profile
            if 'profile' in record:
                profile = record.pop('profile')
                for key, value in profile.items():
                    record[f'profile_{key}'] = value

            # Flatten redrob_signals
            if 'redrob_signals' in record:
                signals = record.pop('redrob_signals')
                for key, value in signals.items():
                    record[f'redrob_signals_{key}'] = value

            # Handle skill_assessment_scores: serialize to JSON string
            if 'skill_assessment_scores' in record:
                record['skill_assessment_scores_json'] = json.dumps(record.pop('skill_assessment_scores'))

            # Handle expected_salary_range_inr_lpa: split into min and max
            if 'expected_salary_range_inr_lpa' in record:
                salary_range = record.pop('expected_salary_range_inr_lpa')
                if '-' in salary_range:
                    min_str, max_str = salary_range.split('-', 1)
                    try:
                        record['expected_salary_min_inr_lpa'] = float(min_str)
                    except ValueError:
                        record['expected_salary_min_inr_lpa'] = None
                    try:
                        record['expected_salary_max_inr_lpa'] = float(max_str)
                    except ValueError:
                        record['expected_salary_max_inr_lpa'] = None
                else:
                    # If no dash, treat as single value? Or set both to same? We'll set both to same.
                    try:
                        val = float(salary_range)
                        record['expected_salary_min_inr_lpa'] = val
                        record['expected_salary_max_inr_lpa'] = val
                    except ValueError:
                        record['expected_salary_min_inr_lpa'] = None
                        record['expected_salary_max_inr_lpa'] = None

            # Leave skills and career_history as-is (they are lists of dicts)
            # No action needed for these.

            parsed_records.append(record)

    # Create DataFrame from parsed records
    df = pl.DataFrame(parsed_records)

    # Save to Parquet
    df.write_parquet(output_path)
    print(f"Saved parsed data to {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    input_file = "data/candidates.jsonl"
    output_file = "output/parsed_candidates.parquet"
    parse_data(input_file, output_file)