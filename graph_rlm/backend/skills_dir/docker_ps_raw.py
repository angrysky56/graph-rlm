import subprocess


def docker_ps_raw() -> str:
    result = subprocess.run(
        ["docker", "ps"], capture_output=True, text=True, timeout=10
    )
    return (
        f"STDOUT: {result.stdout}\\nSTDERR: {result.stderr}\\nRC: {result.returncode}"
    )
