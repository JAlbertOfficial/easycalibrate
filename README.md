# EasyCalibrate

## Simplify Your Calibration Process with Ease.

EasyCalibrate is a powerful application designed to streamline the calibration process in analytical chemistry. Essential for ensuring accurate measurements, calibration can often be complex and time-consuming. EasyCalibrate simplifies this process, making advanced calibration techniques accessible even to those with limited statistical or programming experience.

With EasyCalibrate, users can input nominal x-values (e.g. the concentration of a chemical in a standard), and the corresponding measured responses (e.g. UV absorption values). The app then evaluates whether an ordinary least squares (OLS) regression without weighting is appropriate, checking for the normal distribution of residuals and homoscedasticity. If these conditions are not met, EasyCalibrate recommends a weighted calibration, using relative errors to determine suitable weights.
This calibration approach aims to improve accuracy, particularly at the lower end of the calibration range, thereby enhancing the sensitivity of the calibration in terms of limit of quantification and determination.

Additionally, EasyCalibrate offers an optional feature allowing users to determine matrix effects through matrix-matched calibration. This feature is particularly relevant for methods relying on mass spectrometry detectors (e.g., LC-MS), offering insights into the impact of matrix composition on analytical results.

By generating diagnostic plots and metrics such as residual plots, Breusch-Pagan test statistics, and Shapiro-Wilk test statistics, the app ensures that your calibration model meets rigorous standards. These tools empower users to evaluate the underlying assumptions of their chosen calibration method, identifying and addressing potential issues like heteroscedasticity or non-normality of residuals.

Moreover, EasyCalibrate calculates key goodness-of-fit measures like mean squared errors (RMSE) and R-squared values, as wel las more comprehensive measures such as relative root mean squared error (RRMSE) and root mean squared relative error. These evaluation metrics provide valuable insights into the accuracy and precision of your calibration, guiding you towards optimal performance.

EasyCalibrate determines the calibration's sensitivity in terms of Limit of Detection (LOD) and Limit of Quantification (LOQ). Understanding these lower limits of analyte detection and quantification is vital for method validation and quality control, ensuring your results meet the highest standards.

EasyCalibrate also provides impressive calibration graphs simplifying the interpretation and evaluation of your calibration functions. This feature not only provides a thorough analysis of the calibration process, but also delivers high quality graphics for reports and articles, elevating the presentation of your results.

If you're unfamiliar with statistical terms, diagnostic plots, and other technical jargon, don't worry. Try out EasyCalibrate and compare it with your usual method and witness the improvement in performance and and save your valuable time!




Built with ❤️ by [JAlbertOfficial](https://github.com/JAlbertOfficial)

## What's this?

- `README.md`: This Document! To help you find your way around
- `streamlit_app.py`: The main app that gets run by [`streamlit`](https://docs.streamlit.io/)
- `requirements.txt`: Pins the version of packages needed
- `LICENSE`: Follows Streamlit's use of Apache 2.0 Open Source License
- `.gitignore`: Tells git to avoid comitting / scanning certain local-specific files
- `.streamlit/config.toml`: Customizes the behaviour of streamlit without specifying command line arguments (`streamlit config show`)
- `Makefile`: Provides useful commands for working on the project such as `run`, `lint`, `test`, and `test-e2e`
- `requirements.dev.txt`: Provides packages useful for development but not necessarily production deployment. Also includes all of `requirements.txt` via `-r`
- `pyproject.toml`: Provides a main configuration point for Python dev tools
- `.flake8`: Because `flake8` doesn't play nicely with `pyproject.toml` out of the box
- `.pre-commit-config.yaml`: Provides safeguards for what you commit and push to your repo
- `tests/`: Folder for tests to be picked up by `pytest`

## Local Setup

Assumes working python installation and some command line knowledge ([install python with conda guide](https://tech.gerardbentley.com/python/beginner/2022/01/29/install-python.html)).

```sh
# External users: download Files
git clone git@github.com:JAlbertOfficial/easycalibrate.git

# Go to correct directory
cd easycalibrate

# Run the streamlit app (will install dependencies in a virtualenvironment in the folder venv)
make run
```

Open your browser to [http://localhost:8501/](http://localhost:8501/) if it doesn't open automatically.

### Local Development

The `Makefile` and development requirements provide some handy Python tools for writing better code.
See the `Makefile` for more detail

```sh
# Run black, isort, and flake8 on your codebase
make lint
# Run pytest with coverage report on all tests not marked with `@pytest.mark.e2e`
make test
# Run pytest on tests marked e2e (NOTE: e2e tests require `make run` to be running in a separate terminal)
make test-e2e
# Run pytest on tests marked e2e and replace visual baseline images
make test-e2e-baseline
# After running tests, display the coverage html report on localhost
make coverage
```
## Deploy

For the easiest experience, deploy to [Streamlit Cloud](https://streamlit.io/cloud)

For other options, see [Streamilt deployment wiki](https://discuss.streamlit.io/t/streamlit-deployment-guide-wiki/5099)

## Credits

This package was created with Cookiecutter and the `gerardrbentley/cookiecutter-streamlit` project template.

- Cookiecutter: [https://github.com/audreyr/cookiecutter](https://github.com/audreyr/cookiecutter)
- `gerardrbentley/cookiecutter-streamlit`: [https://github.com/gerardrbentley/cookiecutter-streamlit](https://github.com/gerardrbentley/cookiecutter-streamlit)
