import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

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
#     # "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah",
#     "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline",

# ]

# commands = [
#     "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline",

# ]

# ###################################### Question 5 Commands ###############################################
# gae_lambdas = [0, 0.95, 0.98, 0.99, 1]
# commands = [f"python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda {l} --exp_name lunar_lander_lambda_{l}" for l in gae_lambdas]

####################################### Question 6 Commands ###############################################

commands = [
    f"python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 -rtg --use_baseline -na --batch_size 5000 --exp_name pendulum_default --seed {seed}" for seed in range(1, 6)
    # f"python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 -rtg --use_baseline -na --batch_size 3000 --exp_name pendulum_modified --discount 0.98 -lr 0.01 --seed {seed}" for seed in range(1, 6)
]

# Initialize lists to store data for plotting
data = []

# Run the commands and capture output logs
for i, command in enumerate(commands):
    print(f"RUNNING COMMAND:{command}\n\n\n")
    output = subprocess.check_output(command, shell=True, text=True)
    data.append(output)

# Initialize dictionaries to store training steps and average returns
steps_data = {}
returns_data = {}

# Parse the output logs
for i, output in enumerate(data):
    lines = output.split('\n')
    steps = []
    returns = []
    
    for line in lines:
        if "Eval_AverageReturn" in line:
            match = re.search(r"Eval_AverageReturn : (-?\d+\.\d+)", line)
            if match:
                returns.append(float(match.group(1)))
        if "Train_EnvstepsSoFar" in line:
            match = re.search(r"Train_EnvstepsSoFar : (\d+)", line)
            if match:
                steps.append(float(match.group(1))) # Assuming each step is 1000 iterations
    
    # Store data in dictionaries
    exp_name = f"seed_{i + 1}"
    steps_data[exp_name] = steps
    returns_data[exp_name] = returns

# print("FUCK")
# print(returns_data["cheetah_baseline"])
# print(steps_data["cheetah_baseline"])
# print(len(returns_data["cheetah_baseline"]))

# Create a multi-line chart
# plt.figure(figsize=(10, 6))
# for exp_name, steps in steps_data.items():
#     plt.plot(steps, returns_data[exp_name], label=exp_name)

average_returns = np.mean([returns_data[exp_name] for exp_name in steps_data.keys()], axis=0)

# Create a single line chart for the average returns
plt.figure(figsize=(10, 6))
plt.plot(steps_data['seed_1'], average_returns, label='Average of All Seeds', color='b')

plt.xlabel("Training Steps")
plt.ylabel("Average Return")
plt.legend()
plt.title("Average Return vs. Training Steps")
plt.grid(True)
plt.savefig("default_average_q6_modified_average_return_plot_5.png")
plt.show()