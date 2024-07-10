import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define the circles for each field with colors
ai_circle = patches.Circle((0.4, 0.6), 0.3, facecolor='#d7938e', linewidth=2, label='AI')
ml_circle = patches.Circle((0.4, 0.6), 0.2, facecolor='#92abc0', linewidth=2, label='Machine Learning')
ds_circle = patches.Circle((0.6, 0.6), 0.25, facecolor='#aac8a4', linewidth=2, label='Data Science')
da_circle = patches.Circle((0.7, 0.6), 0.15, facecolor='#bca9c1', linewidth=2, label='Data Analysis')
dm_circle = patches.Circle((0.5, 0.6), 0.1, facecolor='#dab786', linewidth=2, label='Data Mining')

# Add the circles to the axis
ax.add_patch(ai_circle)
ax.add_patch(ml_circle)
ax.add_patch(ds_circle)
ax.add_patch(da_circle)
ax.add_patch(dm_circle)

# Set the limits and title
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.axis('off')

# Add legend
plt.legend(loc='upper right')

# Show the diagram
plt.show()
