import sys, os, traceback

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

with open("train_debug.log", "w", encoding="utf-8") as log_f:
    def log(msg):
        log_f.write(str(msg) + "\n")
        log_f.flush()
        print(msg, flush=True)

    try:
        log("Importing train_dynamic...")
        from scripts.train_dynamic import main
        log("Running main()...")
        main()
        log("main() finished successfully!")
    except Exception as e:
        log("EXCEPTION IN MAIN:")
        traceback.print_exc(file=log_f)
        traceback.print_exc()

