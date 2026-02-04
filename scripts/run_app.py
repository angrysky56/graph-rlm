import sys
import os
import subprocess
import time

def main():
    print("=== Graph RLM Launcher ===")

    # Setup Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)
    os.environ["PYTHONPATH"] = root_dir + ":" + os.environ.get("PYTHONPATH", "")

    # Check for Ollama
    try:
        import httpx
        try:
            httpx.get("http://localhost:11434/api/tags", timeout=1.0)
            print("[+] Ollama detected running.")
        except:
            print("[-] Ollama not running (Local LLM will be unavailable unless started).")
            print("    To install/run Ollama: curl -fsSL https://ollama.com/install.sh | sh")
            print("    (You can still use the app with embedded Graph logic, but generation will fail or require Cloud API key)")
    except ImportError:
        pass

    # Check for Database
    # We don't check for redis explicitly because the app handles fallback.
    print("[+] Database Mode: Auto-Detect (FalkorDB -> NetworkX Fallback)")

    # Launch UI
    print("[+] Launching UI...")
    cmd = [sys.executable, "-m", "graph_rlm.ui.main"]

    # Pass through arguments
    cmd.extend(sys.argv[1:])

    try:
        subprocess.run(cmd, env=os.environ, check=True)
    except KeyboardInterrupt:
        print("\n[!] User interrupted.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Application crashed with code {e.returncode}")

if __name__ == "__main__":
    main()
