# PowerShell helper to prepare a git repo and push to GitHub Pages
# Update $REMOTE_URL to the repository URL you create on GitHub (https://github.com/<user>/<repo>.git)

$REMOTE_URL = 'https://github.com/salomee4444/avalon-interactive.git'
if ([string]::IsNullOrWhiteSpace($REMOTE_URL)) {
    Write-Host "Please edit this file and set `$REMOTE_URL to your GitHub repo URL before running." -ForegroundColor Yellow
    exit 1
}

Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# Ensure index.html exists (copy from the named interactive file if present)
if (Test-Path .\3d_interactive_with_filters.html -PathType Leaf) {
    Copy-Item -Path .\3d_interactive_with_filters.html -Destination .\index.html -Force
    Write-Host "Copied 3d_interactive_with_filters.html -> index.html"
}

git init
git add .
git commit -m "Prepare interactive site for GitHub Pages"
git branch -M main
git remote add origin $REMOTE_URL
git push -u origin main

Write-Host "If the push succeeded, enable GitHub Pages in your repo Settings -> Pages and choose branch 'main' with root '/'." -ForegroundColor Green
