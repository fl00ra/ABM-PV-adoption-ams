import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("amsterdam_buurten.geojson")

gdf["centroid"] = gdf.geometry.centroid
gdf["x"] = gdf.centroid.x
gdf["y"] = gdf.centroid.y

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue')
gdf.centroid.plot(ax=ax, color='red', markersize=5)
for idx, row in gdf.iterrows():
    plt.text(row["x"], row["y"], row["code"], fontsize=6, ha="center")

plt.title("Amsterdam Buurten with Centroids")
plt.axis("off")
plt.tight_layout()
plt.show()

