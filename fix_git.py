import subprocess

script = """
if test "$GIT_AUTHOR_EMAIL" = "srikarkamma09@gmail.com" || test "$GIT_COMMITTER_EMAIL" = "srikarkamma09@gmail.com"
then
    export GIT_AUTHOR_NAME="VONTIPULIPUJYAGANESHKARTHIK"
    export GIT_AUTHOR_EMAIL="vontipulipujya@gmail.com"
    export GIT_COMMITTER_NAME="VONTIPULIPUJYAGANESHKARTHIK"
    export GIT_COMMITTER_EMAIL="vontipulipujya@gmail.com"
fi
"""

cmd = ['git', 'filter-branch', '-f', '--env-filter', script, '--tag-name-filter', 'cat', '--', '--all']

print("Running git filter-branch...")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
