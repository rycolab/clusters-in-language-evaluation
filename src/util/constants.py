import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'webtext': {
        'models': [
            'human',
            'small-117M', 'small-117M-k40',
            'medium-345M', 'medium-345M-k40',
            'large-762M', 'large-762M-k40',
            'xl-1542M', 'xl-1542M-k40',
        ],
        'url': "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/",
    },
}

MODEL_NAMES = {
    'xl-1542M': 'gpt2-xl',
    'large-762M': 'gpt2-large',
    'medium-345M': 'gpt2-medium',
    'small-117M': 'gpt2',
}


BT_SCORES = {
    'sensible': {
        "('gpt2', 'p0.9')": -7.442,
        "('gpt2', 'p1.0')": -37.805,
        "('gpt2-large', 'p0.95')": 8.781,
        "('gpt2-large', 'p1.0')": -7.106,
        "('gpt2-medium', 'p0.9')": -7.293,
        "('gpt2-medium', 'p1.0')": -32.004,
        "('gpt2-xl', 'p0.95')": 31.888,
        "('gpt2-xl', 'p1.0')": 7.753,
        "human": 43.229,
    },
    'interesting': {
        "('gpt2', 'p0.9')": -0.697,
        "('gpt2', 'p1.0')": -15.487,
        "('gpt2-large', 'p0.95')": 6.785,
        "('gpt2-large', 'p1.0')": -1.532,
        "('gpt2-medium', 'p0.9')": -12.824,
        "('gpt2-medium', 'p1.0')": -34.323,
        "('gpt2-xl', 'p0.95')": 23.046,
        "('gpt2-xl', 'p1.0')": 9.529,
        "human": 25.503,
    },
    'human-like': {
        "('gpt2', 'p0.9')": -15.783,
        "('gpt2', 'p1.0')": -27.518,
        "('gpt2-large', 'p0.95')": 12.553,
        "('gpt2-large', 'p1.0')": -6.935,
        "('gpt2-medium', 'p0.9')": -3.429,
        "('gpt2-medium', 'p1.0')": -30.769,
        "('gpt2-xl', 'p0.95')": 15.664,
        "('gpt2-xl', 'p1.0')": 8.966,
        "human": 47.251,
    },
}
