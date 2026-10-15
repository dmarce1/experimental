# CosmicTiger instructions for Claude

Read `AGENTS.md` in full before changing this project. Then read
`docs/claude_handoff.md` for the architecture, numerical contracts, build and
test workflow, current limitations, and additional standing user instructions.
The handoff contains a complete snapshot of the project standing orders, plus
the user's email prohibition, QB4 login budget policy, and C++ include order.

The active project is `/home/dmarce1/workspace/cosmictiger`, a C++20 HPX/MPI
cosmological N-body code with optional CUDA execution. Inspect the actual
checkout, branch, and uncommitted changes before working. Preserve existing
work. Never infer the implemented feature set from old Git history alone.

The application default for `gravity.ghostWidth` is now **1**. Width 2 remains
supported, and old checkpoints without a saved width retain their historical
width 2. CUDA execution switches still default to off in the executable;
the supplied `default.ini` explicitly turns them on.

The user is considering keeping algorithm design and review with Codex while
delegating implementation to Claude. That is a proposed workflow, not an
instruction to start an unspecified implementation. Follow the current task's
scope and acceptance criteria. Use separate worktrees for concurrent editing,
or coordinate a single writer in this directory. All new branch names must use
snake_case, including any namespace component.
