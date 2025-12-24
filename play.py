import argparse
import sys
import torch

# Import project modules
from utils import PPOAgent, get_repo_tree
from rlkit.envs import UnityEnv
from rlkit.utils import SimpleMetricModule

def run_evaluation(args):
    print(f"--- Starting Evaluation: {args.env_name} ({args.algorithm}) ---")
    
    try:
        # 1. Load Agent
        if args.algorithm.lower() == 'ppo':
            print(f"Loading PPO Agent: {args.config_name}/{args.run_name}...")
            agent = PPOAgent(
                environment_name=args.env_name, 
                config_name=args.config_name, 
                run_name=args.run_name
            )
            policy = agent.get_policy_operator()
        else:
            raise ValueError(f"Algorithm '{args.algorithm}' is not supported.")

        # 2. Load Environment
        print(f"Loading Environment: {args.env_name}...")
        env = UnityEnv(
            name=args.env_name, 
            path=args.env_path, 
            graphics=args.graphics, 
            time_scale=1.0 if args.graphics else 20.0,
            seed=args.seed
        )

        # 3. Rollout
        print(f"Running rollout for {args.steps} steps...")
        with torch.no_grad():
            data = env.rollout(args.steps, policy=policy, break_when_any_done=False)

        # 4. Metrics
        print("\n--- Results ---")
        metrics = SimpleMetricModule(mode="approx")(data)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        env.close()

    except Exception as e:
        print(f"\n\033[91m[ERROR] Failed to run experiment: {e}\033[0m")
        print(f"Tip: Use 'python evaluate.py ls {args.env_name}' to see valid options.")
        sys.exit(1)

def main():
    # Handle 'ls' command manually for cleaner syntax
    if len(sys.argv) > 1 and sys.argv[1] == 'ls':
        parser = argparse.ArgumentParser(description="List available options recursively.")
        parser.add_argument("command", type=str, help="The command (ls)")
        parser.add_argument("filters", nargs="*", help="Scope: [env] [algo] [config]")
        args = parser.parse_args()
        
        # Call the utils function with verbose=True to print the tree
        get_repo_tree(filters=args.filters, verbose=True)
        
    else:
        parser = argparse.ArgumentParser(description="Run a Unity Environment with a trained Agent.")
        parser.add_argument("env_name", type=str, help="Name of the Unity Environment (e.g., 3DBall)")
        parser.add_argument("algorithm", type=str, help="Algorithm used (e.g., ppo)")
        parser.add_argument("config_name", type=str, help="Config name (e.g., conf1)")
        parser.add_argument("run_name", type=str, help="Run name (e.g., run1)")
        
        parser.add_argument("--graphics", action="store_true", help="Enable graphics")
        parser.add_argument("--env_path", type=str, default=None, help="Path to local Unity Env executable")
        parser.add_argument("--steps", type=int, default=1000, help="Steps to rollout")
        parser.add_argument("--seed", type=int, default=1, help="Seed")

        args = parser.parse_args()
        run_evaluation(args)

if __name__ == "__main__":
    main()