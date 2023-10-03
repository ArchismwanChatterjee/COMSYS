import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('City.csv')

# Extract the 'column1' and 'column3' columns
new_df = df.iloc[:, [0, 1]]
new = new_df.City.nunique(dropna=True)
print(new)
new = pd.DataFrame(new)

# # Save the extracted columns to a new CSV file
new.to_csv('output.csv', index=False)
