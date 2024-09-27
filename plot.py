import matplotlib.pyplot as plt

# Initialize lists to store the parsed data
times = []
forces_x = []
forces_y = []
forces_z = []
torques_x = []
torques_y = []
torques_z = []

# Read the data from the file
with open("ft_data.txt", "r") as f:
    for line in f:
        # Parse the line using known format
        parts = line.strip().split(", ")
        time_str = parts[0].split(": ")[1]
        force_str = parts[1:4]  # force: x, y, z
        torque_str = parts[4:7]  # torque: x, y, z

        # Append the values to the respective lists
        times.append(float(time_str))
        forces_x.append(float(force_str[0].split(": ")[1]))
        forces_y.append(float(force_str[1]))
        forces_z.append(float(force_str[2]))
        torques_x.append(float(torque_str[0].split(": ")[1]))
        torques_y.append(float(torque_str[1]))
        torques_z.append(float(torque_str[2]))

# Plot force components over time
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(times, forces_x, label='Force X')
plt.plot(times, forces_y, label='Force Y')
plt.plot(times, forces_z, label='Force Z')
plt.title('Force Components Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.grid(True)

# Plot torque components over time
plt.subplot(2, 1, 2)
plt.plot(times, torques_x, label='Torque X')
plt.plot(times, torques_y, label='Torque Y')
plt.plot(times, torques_z, label='Torque Z')
plt.title('Torque Components Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
