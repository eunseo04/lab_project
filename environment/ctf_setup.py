import os
import pathlib
import shutil
import sys
from pathlib import Path

from __init__ import is_pentestperf_available

if is_pentestperf_available():
    import caibench as ptt

def setup_ctf():
    """Setup CTF environment if CTF_NAME is provided

    Supports parallel execution via CTF_INSTANCE_ID environment variable.
    When CTF_INSTANCE_ID is set (e.g., "_1", "_2"), containers will be named
    ctf_target_1, ctf_target_2, etc., and assigned unique IPs on the shared network.
    """
    from mako.template import Template
    from wasabi import color

    ctf_name = os.getenv("CTF_NAME", None)
    if not ctf_name:
        print(color("CTF name not provided, necessary to run CTF", fg="white", bg="red"))
        sys.exit(1)

    instance_id = os.getenv("CTF_INSTANCE_ID", "")
    instance_suffix = f" (Instance {instance_id})" if instance_id else ""

    print(
        color(f"Setting up CTF{instance_suffix}: ", fg="black", bg="yellow")
        + color(ctf_name, fg="black", bg="yellow")
    )

    # Let ctf.py handle container naming and IP assignment based on CTF_INSTANCE_ID
    # Only pass container_name if explicitly overridden
    ctf_kwargs = {
        "subnet": os.getenv("CTF_SUBNET", "192.168.3.0/24"),
    }

    # Only override container_name if CTF_CONTAINER_NAME is explicitly set
    if os.getenv("CTF_CONTAINER_NAME"):
        ctf_kwargs["container_name"] = os.getenv("CTF_CONTAINER_NAME")
    else:
        # Use default that ctf.py will make unique with instance_id
        ctf_kwargs["container_name"] = "ctf_target"

    # Only override IP if CTF_IP is explicitly set and no instance_id
    # (parallel instances auto-assign IPs)
    if os.getenv("CTF_IP") and not instance_id:
        custom_ip = os.getenv("CTF_IP")
        # Validate against reserved IPs (192.168.3.5 is reserved for attacker)
        if custom_ip.endswith(".5"):
            print(
                color(
                    f"WARNING: IP {custom_ip} is reserved for the attacker/agent. "
                    "This may cause conflicts. Consider using a different IP or let the system auto-assign.",
                    fg="yellow",
                    bg="red",
                    bold=True
                )
            )
        ctf_kwargs["ip_address"] = custom_ip

    ctf = ptt.ctf(ctf_name, **ctf_kwargs)  # pylint: disable=I1101  # noqa
    ctf.start_ctf()

    # Only set CAI_ACTIVE_CONTAINER if CTF_INSIDE is true
    if os.getenv("CTF_INSIDE", "true").lower() == "true":
        try:
            import subprocess
            # Use the actual container name (which may include instance_id)
            container_name = ctf.container_name
            # Get the container ID for the specific container
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0 and result.stdout.strip():
                container_id = result.stdout.strip()
                # Set the CTF container as the active container
                os.environ["CAI_ACTIVE_CONTAINER"] = container_id
                print(
                    color(f"CTF container {container_name} ({container_id[:12]}) set as active environment", fg="black", bg="green")
                )
            else:
                print(
                    color(f"Warning: Could not find {container_name} container ID", fg="white", bg="yellow")
                )
        except Exception as e:
            print(
                color(f"Warning: Could not set CTF container as active: {str(e)}", fg="white", bg="yellow")
            )

    # Get the challenge from the environment variable or default to the
    # first challenge
    challenge_key = os.getenv("CTF_CHALLENGE")  # TODO:
    challenges = list(ctf.get_challenges().keys())
    challenge = (
        challenge_key
        if challenge_key in challenges
        else (challenges[0] if len(challenges) > 0 else None)
    )

    # Use the user master template
    template_path = pathlib.Path(__file__).parent.parent / "prompts" / "core" / "user_master_template.md"
    messages = Template(filename=str(template_path)).render(
        ctf=ctf,
        challenge=challenge,
        ip=ctf.get_ip() if ctf else None,
    )

    print(
        color("Testing CTF: ", fg="black", bg="yellow") + color(ctf.name, fg="black", bg="yellow")
    )
    if not challenge_key or challenge_key not in challenges:
        print(
            color(
                "No challenge provided or challenge not found. Attempting to use the first challenge.",
                fg="white",
                bg="blue",
            )
        )
    if challenge:
        print(
            color("Testing challenge: ", fg="white", bg="blue")
            + color(
                "'" + challenge + "' (" + repr(ctf.flags[challenge]) + ")", fg="white", bg="blue"
            )
        )

    return ctf, messages
