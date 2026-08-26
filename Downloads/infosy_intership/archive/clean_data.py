import pandas as pd
import re
import os

# 1. Load raw dataset directly
raw_file = "customer_support_tickets.csv"
df = pd.read_csv(raw_file)

print(f"Original records: {df.shape[0]}")
print(f"Columns: {df.columns.tolist()}")

# 2. Standardize column names (lowercase with underscores)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Handle duplicates
df.drop_duplicates(inplace=True)
if "ticket_id" in df.columns:
    df.drop_duplicates(subset=["ticket_id"], inplace=True)

# 4. Handle missing values
if "resolution" in df.columns:
    df["resolution"] = df["resolution"].fillna("pending_resolution")

# 5. Clean text fields (remove newline breaks, urls, and special chars)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"[\r\n\t]+", " ", text)  # Removes multiline breaks
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    return text.strip()

text_cols = ["ticket_subject", "ticket_description", "resolution"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

# 6. Save cleaned data to a new CSV
output_file = "cleaned_customer_support_tickets.csv"
df.to_csv(output_file, index=False)

print(f"\n Cleaning finished!")
print(f"Cleaned dataset saved as: {output_file}")
print(f"Final shape: {df.shape}")