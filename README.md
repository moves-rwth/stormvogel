# stormvogel 🐦

An interactive approach to probabilistic model checking

Take a look at [the stormvogel documentation!](https://moves-rwth.github.io/stormvogel/)

For developers/mantainers, see the GitHub Wiki.

## Use With Docker

Install `docker`. Run:
```
docker run -it -p 8080:8080 stormvogel/stormvogel
```

## Setup

Install `poetry`. Install dependencies:
```
poetry install
poetry shell
pip install <path to stormpy> # TODO: Package stormpy nicely
pip install .
```
## Testing

Run
```
pytest
```

## Development

Install `pre-commit` hook:
```
pre-commit install
```
