import traceback
import sys
# increase recursion limit to avoid C-extension import recursion issues
sys.setrecursionlimit(20000)
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
print(f"Repository root: {ROOT}")

py_files = list(ROOT.rglob('*.py'))
# restrict to files that contain a test_ function (quick grep)
candidates = []
for p in py_files:
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        continue
    if 'def test_' in text:
        candidates.append(p)

if not candidates:
    print('No candidate test functions found.')
    sys.exit(0)

results = []
for p in candidates:
    rel = p.relative_to(ROOT)
    print('\n--- Running tests in', rel, '---')
    try:
        spec = importlib.util.spec_from_file_location(str(rel).replace('/', '.'), str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f'IMPORT ERROR in {rel}: {e}')
        traceback.print_exc()
        results.append((str(rel), 'IMPORT_ERROR', str(e)))
        continue
    # find functions
    tests = [getattr(mod, name) for name in dir(mod) if name.startswith('test_')]
    if not tests:
        print('No test_ functions found after import.')
        continue
    for fn in tests:
        name = f"{rel}::{fn.__name__}"
        print('Running', name)
        try:
            fn()
            print('PASS', name)
            results.append((name, 'PASS', ''))
        except Exception as e:
            print('FAIL', name)
            traceback.print_exc()
            results.append((name, 'FAIL', str(e)))

print('\n=== Summary ===')
passes = [r for r in results if r[1]=='PASS']
fails = [r for r in results if r[1]=='FAIL']
imports = [r for r in results if r[1]=='IMPORT_ERROR']
print(f'Passed: {len(passes)}, Failed: {len(fails)}, Import errors: {len(imports)}')
if fails:
    print('\nFailures:')
    for f in fails:
        print('-', f[0], f[2])
if imports:
    print('\nImport errors:')
    for i in imports:
        print('-', i[0], i[2])

# exit with non-zero if any failures/import errors
if fails or imports:
    sys.exit(2)
else:
    sys.exit(0)
