import matplotlib.pyplot as plt

episodes = [1, 2, 3]
scores   = [1, 21, 21]

plt.figure(figsize=(8, 4))
plt.plot(episodes, scores, marker='o', linewidth=3, color='#58a6ff', markersize=10)
plt.fill_between(episodes, scores, alpha=0.1, color='#58a6ff')
plt.title("Self-Improvement Over Episodes", fontsize=14)
plt.xlabel("Episode #")
plt.ylabel("Reward Score")
plt.xticks([1, 2, 3])
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("graph.png", dpi=150)
print("graph.png saved!")