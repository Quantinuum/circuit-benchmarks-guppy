





if __name__ == "__main__":

    from pathlib import Path
    import sys
    from textwrap import dedent

    if len(sys.argv) != 2:
        print("Usage: python link_notebooks_folder.py <path_to_notebook_dir>")
        sys.exit(1)

    Path("docs/generated/notebooks").mkdir(parents=False, exist_ok=True)

    stems = []
    for path in Path(sys.argv[1]).glob("*.ipynb"):
        with Path(f"docs/generated/notebooks/{path.stem}.nblink").open("w") as file:
            file.write(f"""{{\n    "path": "../../../notebooks/{path.name}"\n}}""")
            stems.append(path.stem)

    stem_str = ("\n" + " "*11).join(stems)
    with Path("docs/generated/notebooks/index.rst").open("w") as file:
        file.write(dedent(f"""
        Notebooks
        =========

        .. toctree::
           :hidden:
           :maxdepth: 2

           {stem_str}
        """))


    