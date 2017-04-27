from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import config as cf
import tensorflow as tf
import logging

from bayesian_rnn import BayesianRNN
from reader import ptb_raw_data, ptb_iterator


logger = logging.getLogger(__name__)

def get_config(conf):
    if conf == "small":
        return cf.SmallConfig
    elif conf == "medium":
        return cf.MediumConfig
    elif conf == "large":
        return cf.LargeConfig
    elif conf == "titanx":
        return cf.TitanXConfig
    else:
        raise ValueError('did not enter acceptable model config:', conf)


def get_running_avg_loss(name, metric, running_avg_metric, summary_writer, step, decay=0.999):
    """
    Calculate the running average of losses.
    """
    if running_avg_metric == 0:
        running_avg_metric = metric
    else:
        running_avg_metric = running_avg_metric * decay + (1 - decay) * metric
    loss_sum = tf.Summary()
    loss_sum.value.add(tag=name, simple_value=running_avg_metric)
    summary_writer.add_summary(loss_sum, step)
    logger.info("Metric Reported: {} : {}".format(name, running_avg_metric))
    return running_avg_metric


def main(unused_args):

    config = get_config(FLAGS.model_size)
    log_dir = FLAGS.log_dir

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    logger.info("Preprocessing and loading data")
    raw_data = ptb_raw_data(FLAGS.data_path, "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    train_data, val_data, test_data, vocab, word_to_id = raw_data

    logger.info("Building Model")
    model = BayesianRNN(config, is_training=True)
    model.build()

    logger.info("Preparing Savers and Supervisors")
    saver = tf.train.Saver()
    train_summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"))
    val_summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "validation"))
    supervisor = tf.train.Supervisor(
        logdir=log_dir,
        is_chief=True,
        saver=saver,
        summary_op=None,
        save_summaries_secs=60,
        save_model_secs=600,
        global_step=model.global_step)

    sess = supervisor.prepare_or_wait_for_session(config=tf.ConfigProto(allow_soft_placement=True))
    running_avg_loss = 0
    val_running_avg_loss = 0
    step = 0

    train_data_batcher = ptb_iterator(train_data, model.batch_size, model.num_steps)
    val_data_batcher = ptb_iterator(val_data, model.batch_size, model.num_steps)
    reversed_val_data_batcher = ptb_iterator(val_data, model.batch_size, model.num_steps, reverse=True)

    while not supervisor.should_stop() and step < config.max_epoch:

        try:
            inputs, targets = next(train_data_batcher)

        except StopIteration:
            train_data_batcher = ptb_iterator(train_data, model.batch_size, model.num_steps)
            inputs, targets = next(train_data_batcher)

        (summaries, loss, train_step) = model.run_train_step(sess, inputs, targets)

        train_summary_writer.add_summary(summaries, train_step)
        running_avg_loss = get_running_avg_loss("train_loss", running_avg_loss, loss,
                                                train_summary_writer, train_step)

        try:
            inputs, targets = next(val_data_batcher)
        except StopIteration:
            val_data_batcher = ptb_iterator(val_data, model.batch_size, model.num_steps)
            inputs, targets = next(val_data_batcher)
        (val_summaries, val_loss, val_step) = model.run_train_step(sess, inputs, targets)

        train_summary_writer.add_summary(summaries, train_step)
        val_running_avg_loss = get_running_avg_loss("val_loss", val_running_avg_loss, val_loss,
                                                    val_summary_writer, val_step)

        step += 1
        if step % 100 == 0:
            train_summary_writer.flush()
            val_summary_writer.flush()
    supervisor.stop()


if __name__ == '__main__':
    flags = tf.flags

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    flags.DEFINE_string("model_size", "small", "Size of model to train, either small, medium or large")
    flags.DEFINE_string("data_path", os.path.expanduser("~")+'/ptb/', "data_path")
    flags.DEFINE_string("log_dir", "./log", "path to directory for saving tensorboard logs.")
    FLAGS = flags.FLAGS

    from tensorflow.python.platform import flags
    from sys import argv

    flags.FLAGS._parse_flags()
    main(argv)
