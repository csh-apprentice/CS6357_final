import re
import matplotlib.pyplot as plt

# Function to parse log files
def parse_log_file(file_path):
    iterations = []
    loss = []
    loss_ce = []
    loss_dice = []

    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"iteration (\d+) : loss : ([\d.]+), loss_ce: ([\d.]+), loss_dice: ([\d.]+)", line)
            if match:
                iterations.append(int(match.group(1)))
                loss.append(float(match.group(2)))
                loss_ce.append(float(match.group(3)))
                loss_dice.append(float(match.group(4)))

    return iterations, loss, loss_ce, loss_dice

# Parse the three log files
log_files = {
    "Mean Teacher": "mean_teacher.txt",
    "UA Mean Teacher": "UA_mean_teacherlog.txt",
    "U-Net": "U_net_log.txt"
}

parsed_data = {key: parse_log_file(path) for key, path in log_files.items()}

# Plot data
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
titles = ["Total Loss", "Cross-Entropy Loss", "Dice Loss", "Combined Losses"]
colors = ["blue", "orange", "green"]

for i, (title, ax) in enumerate(zip(titles, axs.flatten())):
    for j, (key, (iterations, loss, loss_ce, loss_dice)) in enumerate(parsed_data.items()):
        if title == "Total Loss":
            ax.plot(iterations, loss, label=key, color=colors[j], alpha=0.7)
        elif title == "Cross-Entropy Loss":
            ax.plot(iterations, loss_ce, label=key, color=colors[j], linestyle="--", alpha=0.7)
        elif title == "Dice Loss":
            ax.plot(iterations, loss_dice, label=key, color=colors[j], linestyle="-.", alpha=0.7)
        elif title == "Combined Losses":
            ax.plot(iterations, loss, label=f"{key} - Total Loss", color=colors[j], alpha=0.7)
            ax.plot(iterations, loss_ce, label=f"{key} - Loss CE", color=colors[j], linestyle="--", alpha=0.5)
            ax.plot(iterations, loss_dice, label=f"{key} - Loss Dice", color=colors[j], linestyle="-.", alpha=0.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

plt.suptitle("Loss Visualization Across Logs", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
