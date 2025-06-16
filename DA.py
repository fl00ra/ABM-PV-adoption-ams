import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

df = pd.read_csv("placeholder_cbs_dataset.csv")

plt.figure(figsize=(10, 6))
sns.histplot(df["GESTINKH"], bins=50, kde=True, color="skyblue")
plt.title("Household Income Distribution (GESTINKH)")
plt.xlabel("Annual Gross Income (€)")
plt.ylabel("Count")
plt.yscale("log")
plt.tight_layout()
plt.show()

zonpv_counts = df["ZONPV"].value_counts().rename({0: "Not Adopted", 1: "Adopted"})



fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()
axs[0].pie(zonpv_counts, labels=zonpv_counts.index, autopct='%1.1f%%', startangle=140)
axs[0].set_title("ZONPV Adoption Distribution")

