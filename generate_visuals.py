"""Legacy wrapper for backward compatibility.

Use `python main.py ...` to regenerate canonical visual outputs in results/.
"""

from main import main as pipeline_main


def main() -> None:
    print("[legacy] generate_visuals.py is deprecated. Redirecting to main.py")
    pipeline_main()


if __name__ == "__main__":
    main()
