# Chimera Lab --- Project Handover

**Purpose:** Carry the technical decisions, website architecture,
workflow, and current writing task into a new conversation inside the
existing Chimera Project.

**Last updated:** 16 November 2026\
**Website:** https://chimeralab314.github.io/\
**GitHub repository:**
https://github.com/chimeralab314/chimeralab314.github.io\
**Default branch:** `main`

------------------------------------------------------------------------

## 1. Project purpose and vision

Chimera Lab explores questions at the intersection of **applied
mathematics, machine learning, and science**, combining theory-driven
reasoning with data-driven experiments.

The guiding motto is:

> To create what others can't imagine.

The intended research workflow is:

1.  Formulate a research question.
2.  Develop the mathematical/theoretical foundation.
3.  Design and run a computational experiment.
4.  Publish reproducible code and notebooks on GitHub.
5.  Explain the theory, methods, and findings in a blog article.
6.  Use the results to generate new questions.

The website is an English-language, research-oriented blog. Code and
notebooks generally live in GitHub repositories; the blog explains the
mathematics and experiments and links to the reproducible work.

## 2. Collaboration preferences

The user wants Chat to act as a discussion and brainstorming partner,
not merely a code generator.

-   Use an informal, clear, patient tone.
-   Explain technical decisions with sound reasoning.
-   Give explicit, step-by-step instructions, including the exact file
    or folder to open.
-   Prefer **one action at a time**, especially when guiding work in VS
    Code.
-   Do not guess about the current state of files or the repository. Ask
    for a screenshot or inspect the relevant file when needed.
-   Do not casually rewrite or redesign code that has already been
    approved.
-   Make small, incremental changes and explain their purpose.
-   Treat the actual repository files as the source of truth for
    implementation.
-   When troubleshooting, distinguish confirmed facts from hypotheses.
-   The user is eager to begin writing the first article; avoid
    unnecessary infrastructure work.

## 3. Website and development environment

### Website stack

-   Static site generator: **Jekyll**
-   Hosting: **GitHub Pages**
-   Templates/layouts: **Liquid, HTML**
-   Styling: custom CSS
-   Mathematical typesetting: **MathJax 4**
-   Source control: Git and GitHub

### Repository and local development

-   GitHub repository: `chimeralab314/chimeralab314.github.io`
-   Published website: `https://chimeralab314.github.io/`
-   Main branch: `main`
-   Local project folder (Windows):

``` text
C:\Users\wsnpr\OneDrive\Documents\Virtual Lab\Blog\chimeralab314.github.io
```

-   Local preview URL: `http://127.0.0.1:4000/`
-   Start the local server from the repository folder with:

``` bash
bundle exec jekyll serve --livereload
```

The user works in VS Code on Windows and uses Firefox for browser
preview.

### Important file structure

``` text
chimeralab314.github.io/
├── _data/
│   └── references.yml
├── _includes/
│   ├── figure.html
│   ├── citation.html
│   ├── references.html
│   └── reproducibility.html
├── _layouts/
│   ├── default.html
│   └── post.html
├── _posts/
│   └── 2026-11-03-gradient-descent-and-heavy-ball.md
├── assets/
│   ├── css/
│   │   └── chimera.css
│   └── images/
│       └── articles/
├── .gitignore
├── _config.yml
├── Gemfile
├── Gemfile.lock
├── README.md
└── about.md
```

This is a reference outline, not a guarantee that no other files exist.
Check the repository before editing.

### Important CSS detail

The custom stylesheet is named `assets/css/chimera.css`.

It was deliberately renamed from `assets/css/style.css` because that
path collided with the stylesheet generated/used by the Jekyll Minimal
theme.

In `_layouts/default.html`, the custom stylesheet is linked using:

``` liquid
<link rel="stylesheet"
      href="{{ '/assets/css/chimera.css' | relative_url }}">
```

**Do not rename this stylesheet back to `style.css`.**

### Justified paragraph alignment --- resolved

The following rule is at the bottom of `assets/css/chimera.css`:

``` css
/* Justified alignment for article paragraphs */
.article-content p {
    text-align: justify;
    text-justify: inter-word;
}
```

On 5 November 2026, the user confirmed that: - `chimera.css` exists in
the GitHub repository on `main`. - The rule is present at the end of the
published source file (around lines 421--425 at that time). - The
published article displays justified paragraph alignment.

This issue is resolved. Do not troubleshoot or change it unless a new
problem appears.

## 4. Article features already implemented

### Mathematical typesetting

MathJax 4 is configured. The article system supports: - Inline and
display mathematics. - Automatic equation numbering. - Equation
numbering within sections (where configured). - Equation labels and
cross-references. - Aligned derivations using standard LaTeX structures
such as `equation` and `aligned`.

The user has tested mathematical rendering and equation references.

### Figures

A reusable include exists at `_includes/figure.html`.

It supports: - Figure display. - Automatic figure numbering. -
Captions. - Alt text. - Source attribution. - Optional source links. -
In-text figure cross-references.

### Citations and references

A central bibliography is stored in `_data/references.yml`.

Related includes: - `_includes/citation.html` -
`_includes/references.html`

Posts can specify references through their front matter using
`page.references`.

The user prefers a consistent References section with books, papers, DOI
links, and arXiv links where appropriate.

### Reproducibility

A reusable include exists at `_includes/reproducibility.html`.

It provides a reproducibility section and a clickable link to a notebook
hosted on GitHub.

A previously tested notebook URL is:

https://github.com/wap314/Gradient-Descent-Experiment/blob/main/GD_experiment.ipynb

The published link label is **"View the Jupyter Notebook"**, and the
user confirmed that it opens correctly.

For future articles, link to the relevant notebook/repository rather
than placing large blocks of code in the article unless there is a clear
explanatory reason.

## 5. Article design and editorial decisions

The site design was intentionally made research-oriented, drawing
inspiration from: - Off the Convex Path - Chris Olah's blog

Established preferences: - English-language articles. - Clean,
academic/research-oriented presentation. - Blog name at the upper left;
navigation links such as Blog, About, and Contact at the upper right. -
Article listing format inspired by Off the Convex Path. - Lora is used
for article/body typography. - Article body font size was set to 18px. -
Paragraphs are globally justified through `.article-content p`. - No
reading-time display. - No abstract. - No table of contents. - Article
structure begins with title, keywords, date, and then the
introduction. - Figures and plots from Python experiments are welcome;
the code itself is generally hosted on GitHub.

Do not redesign the site or reintroduce removed elements unless the user
explicitly asks.

## 6. Current first article

**Title:** *On Gradient Based Optimization --- Part 1: Gradient Descent
and Heavy Ball*

**Post file:**

``` text
_posts/2026-11-03-gradient-descent-and-heavy-ball.md
```

The first article is complete, including its conclusion. Its main topics
include:

1. Introduction
2. Gradient Descent
   - Steepest descent and the update rule
   - Stability and step-size restrictions
   - Optimal step size
   - Convergence analysis
   - Numerical experiments, including dimension/runtime scaling
3. Heavy Ball
   - Momentum formulation
   - Behavior near saddle/flat regions
   - Experiments on a non-convex objective with multiple local minima
4. Conclusion
5. Reproducibility
6. References

The article's current front-matter excerpt is:

> An exploration of the mathematical foundations and numerical behaviour
> of gradient descent and heavy ball.

The user has completed the article and approved the conclusion draft.
The article has not yet been committed and pushed to GitHub.

The user determines the scientific content and arguments. Chat should
provide mathematical discussion, feedback, editing, and technical
support when requested, while preserving the user's intended reasoning
and voice.

## 7. Git and publishing workflow

Use this sequence:

1.  Edit the relevant file in VS Code.
2.  Run the local Jekyll server.
3.  Inspect the local site in Firefox.
4.  Check the changes carefully.
5.  Commit the changes with a clear message.
6.  Push/synchronize to GitHub.
7.  Verify the published GitHub Pages website.

The `.gitignore` includes generated/local directories such as `_site/`,
`.jekyll-cache/`, `.sass-cache/`, and `.bundle/`.

GitHub Pages deployment may take some time after pushing. Do not assume
deployment has completed until the published site has been checked.

If a published change does not appear: - First verify the file and
commit on the correct GitHub repository/branch. - Then inspect GitHub
Pages deployment/build status if needed. - Check the published HTML/CSS
resource path. - Consider browser cache only after confirming the
deployed resource exists and contains the expected content.

Do not infer that a change was pushed merely because the user clicked
Synchronize Changes; verify the repository or published result.

## 8. Previous technical research context

The first article belongs to a broader line of work on gradient-based
optimization.

The user has previously implemented gradient descent experiments in
Python notebooks. Topics include: - Quadratic objectives and
positive-definite Hessian matrices. - Eigenvalue-based construction of
test matrices. - Stability and step-size restrictions. - Optimal
step-size behavior. - Convergence rates and condition numbers. - Runtime
scaling with dimension. - Comparisons between Gradient Descent and
Stochastic Gradient Descent. - Heavy Ball / momentum methods.

A prior experiment used a custom eigenvalue method to construct
quadratic problems with controlled eigenvalue spectra. The user planned
to vary dimensions such as `n = 10, 100, 500, 1000, 2000` to investigate
scaling. This is relevant background, but do not assume those
experiments are already part of the first article without checking with
the user.

## 9. Current status as of 16 November 2026

### Completed

-   Website architecture and visual design: established.
-   MathJax and equation features: implemented and tested.
-   Figure numbering/cross-references: implemented and tested.
-   Central references/bibliography: implemented and tested.
-   Reproducibility include and notebook link: implemented and tested.
-   Custom stylesheet path (`chimera.css`): established; do not rename it.
-   First article, including its conclusion: complete.
-   Two temporary posts removed from `_posts`:
    - `2026-10-30-gradient-descent-preview.md`
    - `2026-10-30-mathjax-test.md`
-   Article paragraph line-height changed from `1.75` to `1.4` in
    `assets/css/chimera.css`; the user inspected the local preview and
    said it looks good.
-   About-page paragraph justification added to `assets/css/chimera.css`:

    ```css
    .main-content p {
        text-align: justify;
        text-justify: inter-word;
    }
    ```

-   First article excerpt changed from “computational behavior” to
    “numerical behaviour”.

### Pending

-   Review the current Git changes and commit/push the website changes to
    `chimeralab314/chimeralab314.github.io`.
-   Commit/push the Python notebook changes to its separate notebook
    repository, after inspecting that repository's current Git status.
-   Decide on and implement the article Commentary feature. The exact
    intent is not yet settled: it may be a reader-facing discussion
    system or a static section authored by the user.
-   Test the published blog on a mobile phone, including navigation,
    typography, article layout, and figures.

### Current Git status observed in VS Code

The website repository's Source Control panel showed 21 changes at the
time of the latest inspection. Visible items included:

-   `CHIMERA_HANDOVER.md` (untracked)
-   `_data/references.yml` (modified)
-   The two temporary posts (deleted)
-   The first article (modified)
-   `assets/css/chimera.css` (modified)
-   Multiple new article images under `assets/images/articles/`

The article images appear to be required by the first article and should
be included in the website commit. `CHIMERA_HANDOVER.md` is intentional
project documentation and should also be retained and committed. Confirm
the full change list before staging; do not assume the visible 21 items
are still the current status.

The Python notebook did not appear in the website repository's visible
Source Control list. Its changes are expected to be handled separately
in the notebook repository.

**Recommended immediate task: inspect and stage the intended website
changes, then commit and push the first article and its supporting
assets.**

## 10. Instructions for the next conversation

Please read this handover note as the working project context.

-   Continue from the current pending tasks in Section 9.
-   Work incrementally, one step or issue at a time.
-   Give explicit instructions with exact file paths and ask for
    screenshots or source excerpts when the current state is uncertain.
-   Discuss mathematical correctness and clarity, not just grammar.
-   Preserve the user's voice and intended scientific argument.
-   When code changes are needed, inspect the existing relevant file
    first and propose the smallest safe change.
-   Do not casually alter approved website design or infrastructure.
-   Use the actual repository files and current Git status as the source
    of truth for implementation details.
-   Do not assume that a commit was pushed or that GitHub Pages deployed
    successfully; verify the remote repository and published website.

The first article is complete. The current phase is publishing it,
updating the associated notebook, deciding on the Commentary feature,
and checking the mobile experience before moving on to the next SGD
experiment.
