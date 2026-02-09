import pandas as pd

# Load the CSV, using the first row as header automatically
df = pd.read_csv("zed_pointcloud.csv", header=0)

# Since the columns have no proper names, we can reference by position
# x = column 0, y = column 1
x = df.iloc[:, 0]
y = df.iloc[:, 1]

# Compute distance from (0,0)
distance = (x**2 + y**2)**0.5

# Find the row index with the minimum distance
min_index = distance.idxmin()

print(f"The line number where (x, y) is closest to (0,0) is: {min_index + 2}")
print("Row data:", df.iloc[min_index].to_dict())
