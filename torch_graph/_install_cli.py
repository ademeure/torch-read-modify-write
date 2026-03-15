"""CLI: run any script with torch.compile replaced by aten capture+install.

Usage:
    python -m torch_graph install script.py [script args...]
    python -m torch_graph install -m package.module [script args...]

This patches torch.compile BEFORE running the target script, so every
torch.compile(model) call is intercepted transparently. No changes to
the target script are needed.

Example — run nanochat training with aten graphs instead of Inductor:
    cd nanochat
    python -m torch_graph install -m scripts.base_train --depth=4 --num-iterations=20

Example — capture modded-nanogpt for 3 steps:
    python -m torch_graph install train_gpt.py --max-steps 3
"""

import argparse
import os
import runpy
import sys


def main():
    # We only parse our own flags (before the script path).  Everything
    # after the script path is forwarded as-is to the target script.
    our_args, rest = _split_args(sys.argv[2:])  # sys.argv[0]="torch_graph", [1]="install"

    parser = argparse.ArgumentParser(
        prog="python -m torch_graph install",
        description="Run a script with torch.compile replaced by aten capture+install.",
    )
    parser.add_argument("script", nargs="?", help="Path to Python script")
    parser.add_argument("-m", dest="module", help="Run a module (like python -m)")
    parser.add_argument("--cache-dir", default=".torch_graph_cache",
                        help="Directory for cached aten files (default: .torch_graph_cache)")
    parser.add_argument("--recapture", action="store_true",
                        help="Force re-capture even if cache exists")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress auto_install status messages")
    parser.add_argument("--static", action="store_true",
                        help="Capture with static shapes (batch size etc. will be hardcoded)")
    parser.add_argument("--dynamic", action="store_true",
                        help="(default, kept for backwards compatibility)")
    parser.add_argument("--graph", action="store_true",
                        help="Generate HTML graph visualization for each captured variant")
    parser.add_argument("--record-tensors", action="store_true",
                        help="Record real tensor values inline in the .py file (large models: uses lots of disk)")
    parser.add_argument("--no-record-tensors", action="store_true",
                        help="(default) Don't record real tensor values (kept for backwards compatibility)")
    parser.add_argument("--h5", action="store_true",
                        help="Dump H5 tensor files alongside aten captures (implies --record-tensors)")
    parser.add_argument("--h5-functions", action="store_true",
                        help="Also dump H5 for compiled functions (optimizer steps etc.)")
    parser.add_argument("--no-pt", action="store_true",
                        help="Skip saving .pt tensor files (useful with --h5 to avoid duplication)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Exit after this many training steps (0 = run forever). "
                             "Useful for scripts without early stopping.")
    parser.add_argument("--no-capture-optimizer", action="store_true",
                        help="Don't auto-capture optimizer.step() aten graphs")
    parser.add_argument("--replay-optimizer", action="store_true",
                        help="Replay captured optimizer aten on subsequent step() calls "
                             "(instead of running eagerly)")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="Run N training steps, print loss summary, then exit. "
                             "Implies --replay-optimizer.")
    parser.add_argument("--record-steps", action="store_true",
                        help="Record per-step losses to training_summary.json in cache dir")
    parser.add_argument("--json-ir", action="store_true",
                        help="Save a lossless JSON IR file (.json) alongside each captured aten .py")
    parser.add_argument("--capture-batch-size", type=int, default=0, metavar="N",
                        help="Override batch size during capture (useful to avoid OOM on large models; "
                             "0 = use original batch size)")
    parser.add_argument("--offload-saved", action="store_true",
                        help="Offload saved tensors to CPU during capture to reduce GPU memory usage")
    parser.add_argument("--use-inductor", action="store_true",
                        help="Use Inductor backend instead of aot_eager for capture")
    parser.add_argument("--compile-aten", action="store_true",
                        help="torch.compile the loaded aten forward/backward for Triton fusion")
    args = parser.parse_args(our_args)

    if not args.script and not args.module:
        parser.error("Provide a script path or -m module")

    # ── Patch torch.compile ──────────────────────────────────────────
    import torch_graph.auto_install as ai
    # --verify implies --replay-optimizer and sets exit_after_capture
    replay = args.replay_optimizer or args.verify > 0
    exit_after = args.verify if args.verify > 0 else args.max_steps

    ai.configure(
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
        force_recapture=args.recapture,
        dynamic=not args.static,
        generate_graph=args.graph,
        record_real_tensors=args.record_tensors or args.h5,
        dump_h5=args.h5,
        dump_h5_functions=args.h5_functions,
        skip_pt=args.no_pt,
        exit_after_capture=exit_after,
        capture_optimizer=not args.no_capture_optimizer,
        replay_optimizer=replay,
        verify_steps=args.verify,
        record_steps=args.record_steps,
        save_json_ir=args.json_ir,
        capture_batch_size=args.capture_batch_size,
        offload_saved=args.offload_saved,
        use_inductor=args.use_inductor,
        compile_aten=args.compile_aten,
    )
    # auto_install.patch() is called on import, torch.compile is already patched.

    # ── Run the target ───────────────────────────────────────────────
    if args.module:
        sys.argv = [args.module] + rest
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    else:
        script = os.path.abspath(args.script)
        if not os.path.isfile(script):
            print(f"Error: {script} not found", file=sys.stderr)
            sys.exit(1)
        sys.argv = [script] + rest
        # Add script's directory to sys.path so relative imports work
        script_dir = os.path.dirname(script)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        runpy.run_path(script, run_name="__main__")


def _split_args(argv):
    """Split argv into (our_flags, script_rest).

    Our flags are everything up to and including the script path (or -m module).
    The rest is forwarded to the target script.
    """
    our = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-m":
            # -m module_name — consume both
            our.append(arg)
            if i + 1 < len(argv):
                our.append(argv[i + 1])
                i += 2
            else:
                our.append("")
                i += 1
            break
        elif arg.startswith("-"):
            # Our flag — consume it (and its value if it takes one)
            our.append(arg)
            i += 1
            if arg in ("--cache-dir", "--max-steps", "--verify", "--capture-batch-size") and i < len(argv) and not argv[i].startswith("-"):
                our.append(argv[i])
                i += 1
        else:
            # Positional = script path
            our.append(arg)
            i += 1
            break
    rest = argv[i:]
    return our, rest
