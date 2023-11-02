import subprocess
import re
import matplotlib.pyplot as plt

###################################### Question 3 Commands ###############################################
# Define the commands to run
# commands = [
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole_lb",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_lb_rtg",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_lb_na",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_lb_rtg_na"
# ]

# commands = [
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na",
#     "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na"
# ]

##############################################################################################################

###################################### Question 4 Commands ###############################################
# commands = [
#     "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah"
# ]

commands = [
    "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline",
    "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 3 --exp_name cheetah_baseline_3bgs",
]



# Initialize lists to store data for plotting
data = []

# Run the commands and capture output logs
for command in commands:
    output = subprocess.check_output(command, shell=True, text=True)
    data.append(output)

# Initialize dictionaries to store training steps and average returns
steps_data = {}
returns_data = {}
baseline_losses_data = {}

# Parse the output logs
for i, output in enumerate(data):
    lines = output.split('\n')
    steps = []
    returns = []
    baseline_losses = []

    for line in lines:
        if "Eval_AverageReturn" in line:
            match = re.search(r"Eval_AverageReturn : (-?\d+\.\d+)", line)
            if match:
                returns.append(float(match.group(1)))
        if "Train_EnvstepsSoFar" in line:
            match = re.search(r"Train_EnvstepsSoFar : (\d+)", line)
            if match:
                steps.append(float(match.group(1))) # Assuming each step is 1000 iterations
                # Add parsing for baseline loss here
        if "Baseline Loss" in line:
            match = re.search(r"Baseline Loss : (-?\d+\.\d+)", line)
            if match:
                baseline_losses.append(float(match.group(1)))
    
    # Store data in dictionaries
    exp_name = commands[i].split(" ")[-1]
    steps_data[exp_name] = steps
    returns_data[exp_name] = returns
    baseline_losses_data[exp_name] = baseline_losses


# Create a multi-line chart
plt.figure(figsize=(10, 6))
for exp_name, steps in steps_data.items():
    # plt.plot(steps, returns_data[exp_name], label=f"{exp_name} Eval Return")
    plt.plot(steps, baseline_losses_data[exp_name], label=f"{exp_name} Baseline Loss")

plt.xlabel("Training Steps")
plt.ylabel("Value")
plt.legend()
plt.title("Baseline Loss vs. Training Steps")
plt.grid(True)
plt.savefig("q4_yes_baseline_eval_return_baseline_loss_plot.png")
plt.show()