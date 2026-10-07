#!/usr/bin/env python3
"""
Global floating input box for Linux (X11).

Shows a small always-on-top text field near the top of the screen that
grabs ALL keyboard input system-wide -- no other window or application
can receive keystrokes while it's open. Press Enter to submit (the text
is returned), or Escape to cancel (returns None).

Requires: python3-tk
    Debian/Ubuntu: sudo apt install python3-tk
    Fedora:        sudo dnf install python3-tkinter
    Arch:          sudo pacman -S tk

NOTE: The global grab relies on X11 (Xorg) behavior. Under native Wayland,
compositors deliberately block this kind of system-wide input grab for
security reasons, so it may only capture input to the app itself rather
than truly blocking every other window. It works reliably under Xorg or
XWayland.

Usage as a function (import it):
    from quick_input import get_input
    text = get_input("Type something:")
    if text is not None:
        print("You typed:", text)

Usage as a standalone script (e.g. bound to a hotkey, or called from
another script/shell command -- prints the result to stdout):
    python3 quick_input.py "Your prompt here"
"""

import sys
import tkinter as tk
from typing import Optional


def get_input(prompt: str = "Input:", width: int = 500) -> Optional[str]:
    """
    Opens a floating, always-on-top text box that grabs all keyboard
    input on the system. Blocks until the user presses Enter or Escape.

    Returns the typed string on Enter, or None if cancelled with Escape.
    """
    result = {"value": None}

    root = tk.Tk()
    root.withdraw()  # we don't want a normal root window, just our box

    box = tk.Toplevel(root)
    box.overrideredirect(True)         # no title bar / borders
    box.attributes("-topmost", True)   # stay above every other window

    # Center horizontally, pin near the top of the screen
    screen_w = box.winfo_screenwidth()
    x = (screen_w - width) // 2
    y = 80
    box.geometry(f"{width}x74+{x}+{y}")
    box.configure(bg="#1e1e1e", highlightthickness=2, highlightbackground="#4da3ff")

    label = tk.Label(
        box, text=prompt, fg="#cccccc", bg="#1e1e1e", font=("Sans", 10)
    )
    label.pack(anchor="w", padx=10, pady=(6, 0))

    entry = tk.Entry(
        box, font=("Sans", 14), bg="#2b2b2b", fg="#ffffff",
        insertbackground="#ffffff", relief="flat"
    )
    entry.pack(fill="x", padx=10, pady=(0, 8), ipady=6)

    def cleanup():
        try:
            entry.grab_release()
        except tk.TclError:
            pass
        box.destroy()
        root.quit()

    def submit(event=None):
        result["value"] = entry.get()
        cleanup()

    def cancel(event=None):
        result["value"] = None
        cleanup()

    entry.bind("<Return>", submit)
    entry.bind("<Escape>", cancel)

    box.update_idletasks()
    entry.focus_force()

    # This is the key line: a global grab captures keyboard (and mouse,
    # for this widget) input for the WHOLE X session, not just this app.
    # Every other window is locked 2311649-C/3108788-Mout of keyboard input until released.
    entry.grab_set_global()

    root.mainloop()
    root.destroy()
    return result["value"]


if __name__ == "__main__":
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "Input:"
    text = get_input(prompt_text)
    if text is not None:
        print(text)