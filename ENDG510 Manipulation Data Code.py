import pandas as pd
import random

df = pd.read_csv('data.csv') # Read the CSV file into a DataFrame
i = 1
while i < 200:
    new_data = {
    'Temp': random.randint(-50,50),
    'Humd': random.randint(0,100),
    'Label': 0  }
    df = df.append(new_data, ignore_index=True)
    i+=1

# Step 3: Save the DataFrame to a new CSV file
df.to_csv("data.csv", index=False)
print('done')





