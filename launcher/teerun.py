"""Run a Python module in this console AND copy everything it prints to a log file, so the launcher page can show
the output live while the operator answers the tool's questions in the console window.

usage: python -m launcher.teerun <log file> <module> [module arguments ...]
       python -m launcher.teerun out.log zone.tools.characterize --ip 192.168.0.100 --out reports/x

Used for the PTZ characterization (zone/tools/characterize.py), which asks for confirmation before it moves the
camera and, later, for two marks a known angle apart. Those questions stay in the visible console window; the
page only reads the log. At the end the window waits for Enter, so the summary can be read before it closes.
"""
import runpy
import sys


class Tee:
    """A stream that writes to the console and to a file."""

    def __init__(self, console, log_file):
        self.console, self.log_file = console, log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()          # the page polls the file: write through at once
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    log_path, module, module_args = argv[0], argv[1], argv[2:]
    code = 0
    with open(log_path, "a", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
        sys.argv = [module] + module_args
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except KeyboardInterrupt:
            print("\ninterrompido (Ctrl+C)")
            code = 130
        except BaseException as e:  # noqa: BLE001 (show it in the window and in the log)
            import traceback
            traceback.print_exc()
            code = 1
            del e
        print("\n[fim: código %s]" % code)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        input("Pressione Enter para fechar esta janela. ")
    except (EOFError, KeyboardInterrupt):
        pass
    return code


if __name__ == "__main__":
    sys.exit(main())
