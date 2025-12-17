"""
os



@Author: linlin
@Date: Nov 20 2025
"""

import os
import subprocess
import sys


def open_in_file_manager(path):
    """
    Open the given path in the system's file manager. If the path is a file, it will be highlighted.
    """
    path = os.path.abspath(path)

    if os.path.isfile(path):
        # If it's a file, get its directory:
        file_path = path
        folder_path = os.path.dirname(file_path)
    else:
        # If it's a directory, open it directly:
        file_path = None
        folder_path = path

    if sys.platform.startswith('darwin'):  # macOS
        if file_path:
            subprocess.run(['open', '-R', file_path])  # reveal in Finder
        else:
            subprocess.run(['open', folder_path])
    elif sys.platform.startswith('win'):  # Windows
        if file_path:
            subprocess.run(['explorer', '/select,', file_path])
        else:
            subprocess.run(['explorer', folder_path])
    elif sys.platform.startswith('linux'):  # Linux
        if file_path:
            # Try common file managers to select the file:
            try:
                subprocess.run(['nautilus', '--select', file_path])
            except FileNotFoundError:
                try:
                    subprocess.run(['dolphin', '--select', file_path])
                except FileNotFoundError:
                    subprocess.run(['xdg-open', folder_path])
        else:
            subprocess.run(['xdg-open', folder_path])
    else:
        raise RuntimeError(f'Unsupported OS: {sys.platform}')


def make_open_button(path, label='Open in File Manager'):
    import ipywidgets as widgets
    from IPython.display import display
    button = widgets.Button(description=label, button_style='success')
    button.on_click(lambda b: open_in_file_manager(path))
    display(button)
