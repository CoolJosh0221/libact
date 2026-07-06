project(
    'libact',
    'c',
    'cython',
    'cpp',
    meson_version: '>=1.5.0',
    default_options: ['buildtype=release', 'c_std=c11', 'cpp_std=c++17'],
)
# When neither compiled extension is requested (e.g. the Windows pure-Python
# build, invoked with -Dhintsvm=false -Dvariance_reduction=false), the package
# contains only .py files, so mark the installation pure to produce a
# py3-none-any wheel. The default build (both options true) keeps pure: false
# and ships the compiled extensions on Linux/macOS, unchanged.
have_ext = get_option('hintsvm') or get_option('variance_reduction')
py = import('python').find_installation(pure: not have_ext)

result = run_command(
    py,
    '-c', 'import os; print("True" if os.environ.get("READTHEDOCS") else "False")',
    check: true,
)
is_readthedocs = result.stdout().strip() == 'True'

if not is_readthedocs
    subdir('libact')
endif