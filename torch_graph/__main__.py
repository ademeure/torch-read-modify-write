"""CLI entry point: python -m torch_graph <script.py> | python -m torch_graph dump <script.py>

Subcommands:
  python -m torch_graph <script.py>           Auto-extract aten graphs
  python -m torch_graph dump <script.py>      Dump tensor data
  python -m torch_graph install <script.py>   Run script with torch.compile patched
  python -m torch_graph kbox <h5_file>        Generate kbox test scripts from H5 dump
"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "dump":
    from torch_graph.op_dump import dump_cli
    sys.argv = [sys.argv[0] + " dump"] + sys.argv[2:]
    dump_cli()
elif len(sys.argv) > 1 and sys.argv[1] == "install":
    from torch_graph._install_cli import main as install_main
    install_main()
elif len(sys.argv) > 1 and sys.argv[1] == "kbox":
    from torch_graph.kbox_gen import kbox_cli
    sys.argv = [sys.argv[0] + " kbox"] + sys.argv[2:]
    kbox_cli()
elif len(sys.argv) > 1 and sys.argv[1] == "explain":
    from torch_graph.explain import explain as _explain_fn
    print("Use torch_graph.explain() from Python — CLI explain not yet implemented.")
    sys.exit(1)
else:
    from torch_graph.auto import main
    main()
