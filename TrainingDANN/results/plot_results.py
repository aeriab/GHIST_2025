import matplotlib.pyplot as plt
import pandas as pd

# Read the file, handling spaces or tabs as separators
df = pd.read_csv('sliding_results.txt', sep=r'\s+', engine='python')

plt.figure(figsize=(10, 6))

# Plot Score vs Image_Index
plt.plot(df['Image_Index'], df['Score'], linewidth=0.5)

plt.title('Score per Image Index')
plt.xlabel('Image Index')
plt.ylabel('Score')

# Set the y-axis limit as requested
plt.ylim(top=0.02) 

plt.savefig('score_plot.png')