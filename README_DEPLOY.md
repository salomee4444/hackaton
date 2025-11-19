How to publish the interactive 3D visualization

This folder contains `3d_interactive_with_filters.html` (interactive Plotly visualization).
To host it publicly you can use GitHub Pages, Netlify, or Vercel. The quickest options are listed below.

Recommended: GitHub Pages
1. Rename or copy `3d_interactive_with_filters.html` to `index.html` at repo root.
2. Create a GitHub repository and push the files.
3. Enable GitHub Pages (Settings → Pages) and choose branch `main` with `/ (root)`.
4. Your site will be available at https://<your-username>.github.io/<repo>/

Quick alternative: Netlify drag & drop
1. Go to https://app.netlify.com/ and log in.
2. Drag & drop the folder containing `index.html` (or the single `3d_interactive_with_filters.html`) into the Netlify UI.
3. Netlify will host and give you a public URL immediately.

If you want the exact commands to push from this folder, use the provided `push_github_commands.ps1` script (update REMOTE_URL before running).

Security note: the HTML includes no secrets; do not publish sensitive data. If you need access control, use a platform that supports authentication.

If you want, I can prepare the repo locally and commit — then you can push to GitHub or I can guide you for Netlify.
