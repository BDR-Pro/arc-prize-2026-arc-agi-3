"""Make build_notebook.py strip the GIF gallery (repo-only images)."""
import io

p = '/home/bader/kaggle/arc/scripts/build_notebook.py'
src = io.open(p, encoding='utf-8').read()
if 'See it play' in src:
    print('already patched')
    raise SystemExit
old = '(ROOT / "README.md").read_text()'
new = ('__import__("re").sub(r"## See it play.*?## The problem", '
       '"## The problem", (ROOT / "README.md").read_text(), '
       'flags=__import__("re").DOTALL)')
assert old in src, 'marker not found'
src = src.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8').write(src)
print('patched')
