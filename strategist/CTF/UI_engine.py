from engine import (CTFConfig, CTFEnvironment, CTFAction, default_red_actions,)

# ============================================================
# Display
# ============================================================

def print_state(env: CTFEnvironment):
    print("\n" + "=" * 70)
    print("CURRENT CTF STATE")
    print("=" * 70)

    state = env.state

    print(f"Target IP          : {state.target_ip}")
    print(f"Target OS          : {state.target_os}")
    print(f"Attacker IP        : {state.attacker_ip}")
    print(f"Attacker OS        : {state.attacker_os}")

    print(f"\nHosts              : {state.hosts}")
    print(f"Services           : {state.services}")
    print(f"Vulnerabilities    : {state.vulnerabilities}")
    print(f"Credentials        : {state.credentials}")

    print(f"\nCompromised        : {state.compromised}")
    print(f"Privilege          : {state.privilege_level}")
    print(f"Internal Access    : {state.internal_access}")
    print(f"Detected           : {state.detected}")

    print(
        f"\nCompleted Objectives: "
        f"{state.completed_objectives}"
    )

    print(f"Current Objective  : {state.current_objective}")
    print(f"Reward             : {state.reward}")
    print(f"Done               : {state.done}")

    print("=" * 70)


# ============================================================
# Available Actions
# ============================================================

def print_available_actions(env: CTFEnvironment):
    actions = default_red_actions(env.state)

    if not actions:
        print("\nNo predefined actions available.")
        return

    print("\nAvailable Red Actions")
    print("-" * 70)

    for index, action in enumerate(actions):
        print(
            f"[{index}] "
            f"{action.tool:<10} "
            f"{action.description}"
        )
        print(f"     {action.command}")


# ============================================================
# Manual Action
# ============================================================

def execute_manual_action(env: CTFEnvironment):
    print("\nEnter command to execute inside target container.")

    command = input("ctf> ").strip()

    if not command:
        return

    action = CTFAction(
        tool="manual",
        command=command,
        description="Manual CTF command",
        actor="red",
    )

    state, result = env.step(action)

    print("\n" + "=" * 70)
    print("COMMAND RESULT")
    print("=" * 70)

    print(f"Command     : {result.command}")
    print(f"Return code : {result.return_code}")

    print("\n--- STDOUT ---")
    print(result.stdout)

    print("\n--- STDERR ---")
    print(result.stderr)

    print("\n--- STATE ---")
    print(state.get_state_description())


# ============================================================
# Execute Predefined Action
# ============================================================

def execute_predefined_action(env: CTFEnvironment):
    actions = list(default_red_actions(env.state))

    if not actions:
        print("No predefined actions.")
        return

    print_available_actions(env)

    try:
        index = int(input("\nSelect action number: "))
    except ValueError:
        print("Invalid number.")
        return

    if index < 0 or index >= len(actions):
        print("Invalid action index.")
        return

    action = actions[index]

    print("\nExecuting:")
    print(action.command)

    state, result = env.step(action)

    print("\n--- STDOUT ---")
    print(result.stdout)

    print("\n--- STDERR ---")
    print(result.stderr)

    print("\n--- STATE AFTER ACTION ---")
    print(state.get_state_description())


# ============================================================
# Simulation
# ============================================================

def simulate_action(env: CTFEnvironment):
    """
    Run an action through the virtual simulator.

    IMPORTANT:
        This does NOT touch Docker.
    """

    actions = list(default_red_actions(env.state))

    if not actions:
        print("No actions available.")
        return

    print_available_actions(env)

    try:
        index = int(input("\nSelect action number: "))
    except ValueError:
        print("Invalid number.")
        return

    if index < 0 or index >= len(actions):
        print("Invalid action index.")
        return

    action = actions[index]

    simulated_state, reward = (
        CTFEnvironment.simulated_transition(
            env.state,
            action,
        )
    )

    print("\n" + "=" * 70)
    print("SIMULATED RESULT")
    print("=" * 70)

    print(f"Action : {action}")
    print(f"Reward : {reward}")

    print("\n--- SIMULATED STATE ---")
    print(simulated_state.get_state_description())

    print(
        "\nNOTE: "
        "The real Docker container was NOT modified."
    )


# ============================================================
# Main
# ============================================================

def main():

    config = CTFConfig(
        container_name="bank-web",
        target_ip="192.168.3.2",
        target_os="linux",
        attacker_ip="192.168.3.10",
        attacker_os="kali",
        docker_network="financial_cyber_range",
    )

    env = CTFEnvironment(config)

    print("=" * 70)
    print("Financial Cyber Range - CTF Environment")
    print("=" * 70)

    print(
        "\nTarget container:",
        config.container_name,
    )

    print(
        "Target IP:",
        config.target_ip,
    )

    print(
        "\nCommands:"
    )

    print("  observe      - observe Docker target")
    print("  state        - show current CTF state")
    print("  actions      - show predefined Red actions")
    print("  run          - execute predefined action")
    print("  manual       - execute manual command")
    print("  simulate     - simulate action without Docker")
    print("  reset        - reset logical state")
    print("  quit         - exit")

    while True:
        try:
            command = input("\nctf> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if command == "quit":
            break

        elif command == "observe":
            observation = env.observe()
            print("\n--- OBSERVATION ---")
            print(observation)

        elif command == "state":
            print_state(env)

        elif command == "actions":
            print_available_actions(env)

        elif command == "run":
            execute_predefined_action(env)

        elif command == "manual":
            execute_manual_action(env)

        elif command == "simulate":
            simulate_action(env)

        elif command == "reset":
            env.reset()
            print("\nLogical CTF state has been reset.")

        elif command == "":
            continue

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()