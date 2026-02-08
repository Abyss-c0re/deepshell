import os

EXECUTABLE_NAME = "deepshell"


def create_symlink():
    """
    Create symlink for deepshell in ~/.local/bin
    """
    bin_dir = os.path.expanduser("~/.local/bin")
    symlink_path = os.path.join(bin_dir, EXECUTABLE_NAME)

    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)

    if not os.path.exists(symlink_path):
        os.symlink(os.path.abspath(EXECUTABLE_NAME), symlink_path)
        print(f"Symlink created at {symlink_path}")
    else:
        print(f"Symlink already exists at {symlink_path}")


def remove_symlink():
    """
    Remove the symlink for deepshell from ~/.local/bin
    """
    bin_dir = os.path.expanduser("~/.local/bin")
    symlink_path = os.path.join(bin_dir, EXECUTABLE_NAME)

    if os.path.exists(symlink_path):
        os.remove(symlink_path)
        print(f"Symlink removed from {symlink_path}")
    else:
        print(f"No symlink found at {symlink_path}")
