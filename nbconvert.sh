poetry run jupyter nbconvert --execute --to notebook --inplace "docs/**/**.ipynb"  --ExecutePreprocessor.kernel_name=python3
# Useful for testing nbconvert (part of the pipeline) locally
