import os
from pathlib import Path
from sphinx import addnodes
from sphinx.util.typing import ExtensionMetadata
from sphinx.application import Sphinx

def setup(app:Sphinx)->ExtensionMetadata:
    def update_toctree(app, doctree, docname):
            if docname != 'index':
                return

            node = doctree.traverse(addnodes.toctree)[0]
            toc = app.env.resolve_toctree(docname, app.builder, node)

            # do something with "toc" here

    app.connect('doctree-resolved', update_toctree)

    return {
        'version': '0.1',
        'parallel_read_safe': False,
        'parallel_write_safe': False,
    }

def crawl_autosummary_shorten_titles(*args):
    path = f"{Path(__file__).parent.parent}/_autosummary"
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            continue
        partial, extension = os.path.splitext(file_path)
        if extension == ".rst":
            new_path = '.'.join(partial.split('.')[1:])
            new_path = f"{path}/{new_path}{extension}"
            os.rename(file_path, new_path)