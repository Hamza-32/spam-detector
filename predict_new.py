"""Legacy wrapper for backward compatibility.

Use `python predict.py ...` for the canonical prediction entry point.
"""

from predict import main as predict_main


def main() -> None:
    print("[legacy] predict_new.py is deprecated. Redirecting to predict.py")
    predict_main()


if __name__ == "__main__":
    main()
