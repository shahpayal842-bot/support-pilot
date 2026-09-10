import pandas as pd

# Load your dataset (use your cleaned or raw file path)
df = pd.read_csv("cleaned_customer_support_tickets.csv")

# Get rows and columns using .shape
num_rows, num_cols = df.shape

print(f"Total Rows: {num_rows}")
print(f"Total Columns: {num_cols}")

# Condition check
if num_rows < 500 and num_cols < 500:
    print("👉 Yes! Both rows and columns are below 500.")
else:
    print("👉 No, at least one of them (or both) is 500 or higher.")