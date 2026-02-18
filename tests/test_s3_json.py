import pandas as pd

df = pd.read_csv("data/raw/abalone.csv.txt")

df.to_json("data/raw/abalone.json", orient="records", lines=True)
df.to_csv("data/raw/abalone.txt", sep="\t", index=False)

print("done")
