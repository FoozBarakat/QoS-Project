import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(10, 6))

# Ideal clock
ideal_times = [10, 20, 30, 40, 50]
ideal_labels = ['Packet'] * len(ideal_times)

# Clock with jitter
jitter_times = [10, 19, 29, 40, 50]
jitter_labels = ['Packet'] * len(jitter_times)

# Plot ideal clock
plt.plot(ideal_times, [0.6]*len(ideal_times), 'bx-', label='Ideal Clock')
for i, txt in enumerate(ideal_labels):
    plt.annotate(txt, (ideal_times[i], 0.6), textcoords="offset points", xytext=(0,-15), ha='center')

# Plot clock with jitter
plt.plot(jitter_times, [0.4]*len(jitter_times), 'rx-', label='Clock with Jitter')
for i, txt in enumerate(jitter_labels):
    plt.annotate(txt, (jitter_times[i], 0.4), textcoords="offset points", xytext=(0,10), ha='center')

# Add labels and title
plt.yticks([0.4, 0.6], ['Clock with Jitter', 'Ideal Clock'])
plt.xlabel('Time (ms)')
plt.title('Ideal Clock vs Clock with Jitter')
plt.grid(True)

# Show the plot
plt.show()
