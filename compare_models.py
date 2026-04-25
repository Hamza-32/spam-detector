"""Legacy wrapper for backward compatibility.

Use `python main.py ...` for the canonical pipeline.
"""

from main import main as pipeline_main


def main() -> None:
    print("[legacy] compare_models.py is deprecated. Redirecting to main.py")
    pipeline_main()


if __name__ == "__main__":
    main()
