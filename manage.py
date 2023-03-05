import sys
import subprocess


COMMANDS = {
    "check": ["pylama", "-l", "pylint", "-i", "D100,D103"],
    "test": ["python", "-m", "pytest"],
    "run": ["python", "-m", "src.main"],
}


def main():
    if len(sys.argv) != 2:
        print("Available commands:\n\t{}".format(
            '\n\t'.join(COMMANDS)
        ))
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        sys.exit(2)
    subprocess.run(COMMANDS[cmd], shell=True)


if __name__ == "__main__":
    main()
