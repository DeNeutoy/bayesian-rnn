
def get_config(conf):
    if conf == "small":
        return SmallConfig
    elif conf == "medium":
        return MediumConfig
    elif conf == "large":
        return LargeConfig
    else:
        raise ValueError('did not enter acceptable model config:', conf)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    learning_rate_decay = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 200
    embedding_size = 200
    hidden_size = 20
    max_epoch = 1000000
    batch_size = 20
    vocab_size = 10000
    summary_frequency = 10


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    learning_rate_decay = 0.8
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    embedding_size = 650
    hidden_size = 650
    max_epoch = 1000000
    batch_size = 32
    vocab_size = 10000
    summary_frequency = 10


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.0005
    learning_rate_decay = 1/1.15
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    embedding_size = 300
    hidden_size = 1500
    max_epoch = 1000000
    batch_size = 32
    vocab_size = 10000
    summary_frequency = 10
