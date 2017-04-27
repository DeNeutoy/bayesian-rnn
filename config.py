
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    embedding_size = 100
    hidden_size = 200
    max_epoch = 1000000
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.0005
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    embedding_size = 300
    hidden_size = 650
    max_epoch = 1000000
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.0005
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    embedding_size = 300
    hidden_size = 1500
    max_epoch = 1000000
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TitanXConfig(object):
    """For Titan X -- Faster Training"""
    init_scale = 0.04
    learning_rate = 0.0005
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    embedding_size = 300
    hidden_size = 1500
    max_epoch = 1000000
    lr_decay = 1 / 1.10
    batch_size = 64
    vocab_size = 10000
