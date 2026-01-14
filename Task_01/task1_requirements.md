IMPLEMENT TASK 1: Iris Dataset Exploration & Visualization

Phase: Implementation (Phase 1 — Specification already complete)

Goal:
Create a complete Jupyter Notebook named `Task1_Iris_Exploration.ipynb` that performs exploratory data analysis and visualization of the Iris dataset. The notebook must be fully runnable and include Markdown explanations and commented Python code.

Requirements (strict):
1. Environment / Libraries:
   - Use Python 3.x with these libraries: pandas, numpy, matplotlib, seaborn, os.
   - Import libraries at the top and show their versions in a small cell.

2. Data Loading:
   - Load the Iris dataset using `sns.load_dataset('iris')`.
   - If `sns` load fails, fallback to `pd.read_csv('iris.csv')` (but prefer sns).

3. Data Inspection:
   - Print and display: `df.shape`, `df.columns`, `df.head()`, `df.info()`, `df.describe()`.
   - Add a short Markdown note about missing values or data cleanliness. If none, explicitly state "No missing values detected."

4. Visualizations (each must be plotted inline AND saved as PNG into `./figures/`):
   - Create directory `./figures/` if it does not exist.
   - Scatter plots:
     - `sepal_length` vs `sepal_width` colored by `species`.
     - `petal_length` vs `petal_width` colored by `species`.
   - Pairplot of all numeric features colored by `species` (seaborn `pairplot`).
   - Histograms for each numeric feature (one figure per feature OR a combined multi-panel figure).
   - Box plots for each numeric feature (one panel per feature).
   - For each figure: include a descriptive title, axis labels, and legend (if applicable). Save each figure with a descriptive filename in `./figures/` (e.g., `figures/iris_sepal_scatter.png`).

5. Code quality:
   - Organize code into clearly separated cells: imports, data loading, inspection, plotting functions, plotting cells, observations.
   - Use descriptive variable names and comments explaining each major step.
   - Use functions where appropriate for repeated plotting logic.

6. Explanations & Insights:
   - After visualizations, add a Markdown cell with **3–6 concise observations** that reference the figures/statistics (e.g., which features separate species well, presence of outliers).
   - Add a short "Next steps" bullet list suggesting modeling approaches or preprocessing for future tasks.

7. Outputs & Saving:
   - Ensure every saved figure file is created in `./figures/`.
   - Optionally include a final cell that prints a short submission checklist confirming all required outputs are present.

8. Deliver final artifacts:
   - The notebook content (full Markdown + code cells).
   - A short `task1_README.md` (1–2 paragraphs summarizing objective, dataset, key findings, and files produced).
   - A final message (in the notebook or as Claude output) confirming completion and listing produced files.

Formatting expectations for the notebook:
- Use Markdown headings before each section.
- Use fenced python code blocks in the notebook cells.
- Keep Markdown explanations concise and technical.

Produce the **complete notebook content** (Markdown and code cells) as the response. If you can write an `.ipynb` file directly, do so; otherwise provide the full notebook content ready to copy into a Jupyter Notebook file. Also produce `task1_README.md` content. End the response with a one-paragraph summary of key insights found.

Begin implementation now and output the notebook and README content in your response.
