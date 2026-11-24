import sqlite3
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")

    import jax
    print(f"JAX: {jax.__version__} | Devices: {jax.devices()}")

    import pymdp
    print(f"pymdp: {getattr(pymdp, '__version__', 'installed (JAX backend)')}")

    import ribs
    print(f"pyribs: {ribs.__version__}")

    import sqlite_vec
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    vec_ver = conn.execute("select vec_version()").fetchone()[0]
    print(f"sqlite-vec: {vec_ver}")

    import fastembed
    print(f"fastembed: {getattr(fastembed, '__version__', 'installed')}")

    try:
        import ollama
        models = [m.model for m in ollama.list().models]
        print(f"Ollama reachable | Models: {models}")
    except Exception as exc:
        print(f"Ollama check skipped/unreachable: {exc}")

    print("\nSprint 0 environment check PASSED.")


if __name__ == "__main__":
    main()