import pandas as pd

# 1. Load your full cleaned dataset
df = pd.read_csv("cleaned_customer_support_tickets.csv")

# 2. Take a smaller random sample of 500 rows
df_sample = df.sample(n=500, random_state=42)

# 3. Save it back with the same name so it replaces the large file
df_sample.to_csv("cleaned_customer_support_tickets.csv", index=False)

print(f"Sampled dataset created successfully! New shape: {df_sample.shape}")