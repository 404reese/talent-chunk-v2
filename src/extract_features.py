import polars as pl
from sentence_transformers import SentenceTransformer

def construct_profile_text(row):
    """
    Constructs a full profile text representation from a parsed candidate row.
    """
    text_parts = []
    
    if row.get('profile_name'):
        text_parts.append(f"Name: {row.get('profile_name')}")
    if row.get('profile_age'):
        text_parts.append(f"Age: {row.get('profile_age')}")
    if row.get('profile_location'):
        text_parts.append(f"Location: {row.get('profile_location')}")
    
    # Skills
    skills = row.get('skills', [])
    if skills:
        skill_texts = [f"{s.get('name', '')} ({s.get('level', '')})" for s in skills]
        text_parts.append("Skills: " + ", ".join(skill_texts))
        
    # Career History
    career = row.get('career_history', [])
    if career:
        career_texts = [f"{c.get('role', '')} at {c.get('company', '')} for {c.get('years', '')} years" for c in career]
        text_parts.append("Career History: " + "; ".join(career_texts))
        
    return "\n".join(text_parts)

def extract_features(input_path: str, output_path: str) -> None:
    print(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)
    
    # Construct full profile text
    print("Constructing full profile text for candidates...")
    texts = [construct_profile_text(row) for row in df.to_dicts()]
    
    # Load BGE-M3 model
    print("Loading BGE-M3 model (BAAI/bge-m3)...")
    # This model outputs 1024-dimensional dense vectors
    model = SentenceTransformer("BAAI/bge-m3")
    
    # Generate embeddings
    print("Generating dense vector embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add embeddings as a new column (list of floats)
    print("Saving features...")
    df_features = df.with_columns(
        pl.Series("semantic_vector", embeddings.tolist())
    )
    
    df_features.write_parquet(output_path)
    print(f"Saved candidate features to {output_path}")
    print(f"New shape: {df_features.shape}")

if __name__ == "__main__":
    input_file = "output/parsed_candidates.parquet"
    output_file = "output/semantic_features.parquet"
    extract_features(input_file, output_file)
