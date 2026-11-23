"""CAGE4 evaluator arithmetic; shared rewards are NEVER summed over Blue agents."""
from statistics import mean, stdev


class JointRewardLedger:
    def __init__(self, profile="official"):
        if profile not in ("official", "terminal_inclusive"):
            raise ValueError("Unknown scoring profile")
        self.profile = profile
        self.rewards = []
        self.raw_rewards = []

    def add(self, rewards, terminal):
        values = [float(v) for k, v in rewards.items() if k != "__all__"]
        if not values:
            raise ValueError("Missing environment reward")
        if max(values) - min(values) > 1e-7:
            raise ValueError("Expected duplicated shared Blue reward, received different agent values")
        joint = mean(values)
        self.raw_rewards.append(joint)
        included = not terminal or self.profile == "terminal_inclusive"
        if included:
            self.rewards.append(joint)
        return joint, included

    @property
    def total(self):
        return sum(self.rewards)


def aggregate_episodes(episodes):
    if not episodes:
        return {"n": 0, "mean_joint_reward": None, "reward_std": None, "execution_time_s": None}
    rewards = [e["joint_return"] for e in episodes]
    return {"n": len(rewards), "mean_joint_reward": mean(rewards),
            "reward_std": stdev(rewards) if len(rewards) > 1 else None,
            "execution_time_s": sum(e["execution_time_s"] for e in episodes),
            "mean_episode_time_s": mean(e["execution_time_s"] for e in episodes)}
