# To Do

## Before next merge
~~1. DatafileHandler~~
    ~~1. add task handler/queue to DatafileHandler~~
    ~~1. include validate in DatafileHandler?~~
1. response_models
    1. resolve naming issues between machine learning/pytorch models and pydantic models
    1. rename response_models?
    1. separate response_models internal pydantic models?
    1. start documentation for site users

## Other necessary changes
1. identify "best" trained ML models of 3 from paper
1. Documentation
    1. update documentation in evaluate and analysis
    1. add documentation in datafile, globals, and main
    1. add documentation in response_models?

## Internal design preferences
1. preprocessing
    1. reduce code duplication by sending pre-computed arrays to fit functions
    1. separate preprocessing functions for easier understanding
    1. change Avogadro constant to 0.6022... to simplify conversion
    1. rename GOOD_EXP to NU for math