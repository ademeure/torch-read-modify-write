#!/usr/bin/env python3
"""Generate kernelbox test scripts from torch-graph H5 files.

Usage:
  python h5_to_kbox.py nanogpt.h5 --list                                   # list groups
  python h5_to_kbox.py nanogpt.h5 --list --section backward                # list backward groups
  python h5_to_kbox.py nanogpt.h5 --list --strategy module                 # list module groups
  python h5_to_kbox.py nanogpt.h5 --group 5 --out test.py                  # single group by index
  python h5_to_kbox.py nanogpt.h5 --group "005*" --out test.py             # by glob pattern
  python h5_to_kbox.py nanogpt.h5 --all --out-dir tests/                   # all groups
  python h5_to_kbox.py nanogpt.h5 --section forward --out test_forward.py  # whole section
  python h5_to_kbox.py nanogpt.h5 --section forward --strategy module --all --out-dir tests/
"""

import argparse
import fnmatch
import sys

from torch_graph.kbox_gen import (
    generate_all,
    generate_group_script,
    generate_section_script,
    list_groups,
)


def main():
    p = argparse.ArgumentParser(
        description="Generate kernelbox test scripts from torch-graph H5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("h5", help="Path to the H5 file")
    p.add_argument("--list", action="store_true", help="List available groups")
    p.add_argument("--group", help="Group selector: integer index or glob pattern on group name")
    p.add_argument("--all", action="store_true", help="Generate test files for all groups")
    p.add_argument("--section", choices=["forward", "backward"], help="Section to use (default: both for --list, forward for generation)")
    p.add_argument("--strategy", default="line", choices=["line", "module"], help="Grouping strategy (default: line)")
    p.add_argument("--out", help="Output file path (for --group or --section without --all)")
    p.add_argument("--out-dir", help="Output directory (for --all)")

    args = p.parse_args()

    # ── List mode ──
    if args.list:
        groups = list_groups(args.h5, section=args.section, strategy=args.strategy)
        if not groups:
            print("No groups found.")
            return

        for g in groups:
            n_in = len(g.input_nodes)
            n_out = len(g.output_nodes)
            mod = f" [{g.module_path}]" if g.module_path else ""
            print(f"  {g.index:3d}  {g.section}/{g.strategy}/{g.name}")
            print(f"       {n_in} inputs, {n_out} outputs, {len(g.all_node_names)} ops{mod}")
        print(f"\n  {len(groups)} groups total")
        return

    # ── Generate all ──
    if args.all:
        out_dir = args.out_dir or "kbox_tests"
        section = args.section  # None = both
        paths = generate_all(args.h5, out_dir, section=section, strategy=args.strategy)
        print(f"Generated {len(paths)} test files in {out_dir}/")
        for p_ in paths:
            print(f"  {p_}")
        return

    # ── Generate single group ──
    if args.group is not None:
        section = args.section  # None = both
        groups = list_groups(args.h5, section=section, strategy=args.strategy)

        # Try integer index first
        try:
            idx = int(args.group)
            matches = [g for g in groups if g.index == idx]
            # If ambiguous across sections, prefer forward
            if len(matches) > 1:
                fwd = [g for g in matches if g.section == "forward"]
                if fwd:
                    matches = fwd[:1]
        except ValueError:
            # Glob match on group name
            pattern = args.group
            matches = [g for g in groups if fnmatch.fnmatch(g.name, pattern)]

        if not matches:
            print(f"No groups matching '{args.group}'", file=sys.stderr)
            print("Available groups:", file=sys.stderr)
            for g in groups[:10]:
                print(f"  {g.index}: {g.name}", file=sys.stderr)
            if len(groups) > 10:
                print(f"  ... ({len(groups)} total)", file=sys.stderr)
            sys.exit(1)

        if len(matches) > 1:
            print(f"Multiple matches for '{args.group}':", file=sys.stderr)
            for g in matches:
                print(f"  {g.index}: {g.name}", file=sys.stderr)
            print("Use a more specific pattern or integer index.", file=sys.stderr)
            sys.exit(1)

        group = matches[0]
        out = args.out or f"test_{group.index:03d}.py"
        script = generate_group_script(args.h5, group, out_path=out)
        print(f"Generated {out} for: {group.name}")
        return

    # ── Generate whole section ──
    if args.section and not args.all and not args.group:
        out = args.out or f"test_{args.section}.py"
        script = generate_section_script(args.h5, args.section, args.strategy, out_path=out)
        n_groups = len(list_groups(args.h5, section=args.section, strategy=args.strategy))
        print(f"Generated {out} for {args.section}/{args.strategy} ({n_groups} groups)")
        return

    p.print_help()


if __name__ == "__main__":
    main()
