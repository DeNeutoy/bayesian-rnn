from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import logging

from bayesian_rnn import BayesianRNN
from reader import ptb_raw_data, Batcher
from config import get_config

logger = logging.getLogger(__name__)


def get_running_avg_loss(name, metric, running_avg_metric, summary_writer, step, decay=0.999):
    """
    Calculate the running average of losses.
    """
    if running_avg_metric == 0:
        running_avg_metric = metric
    else:
        running_avg_metric = running_avg_metric * decay + (1 - decay) * metric
    loss_sum = tf.Summary()
    loss_sum.value.add(tag="loss", simple_value=running_avg_metric)
    summary_writer.add_summary(loss_sum, step)
    logger.info("Metric Reported: {} : {}".format(name, running_avg_metric))
    return running_avg_metric


def run_step(name, batcher, step_function, session, summary_writer, running_avg_metric, step, state, memory):

    try:
        inputs, targets = next(batcher)
    except StopIteration:
        batcher.refresh_generator()
        inputs, targets = next(batcher)

    summaries, loss, state, memory, step = step_function(session, inputs, targets,state, memory, step)

    if summaries is not None:
        summary_writer.add_summary(summaries, step)
    running_avg_loss = get_running_avg_loss(name + "_loss", running_avg_metric, loss,
                                            summary_writer, step)
    return running_avg_loss, state, memory


def main(unused_args):

    config = get_config(FLAGS.model_size)
    log_dir = FLAGS.log_dir

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    logger.info("Preprocessing and loading data")
    raw_data = ptb_raw_data(FLAGS.data_path, "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    train_data, val_data, test_data, vocab, word_to_id = raw_data

    if FLAGS.test:
        config.batch_size = 1
        config.num_steps = 1

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
    state = np.zeros([model.batch_size, model.hidden_size]).astype(np.float32)
    memory = np.zeros([model.batch_size, model.hidden_size]).astype(np.float32)

    if FLAGS.test:
        test_data_batcher = Batcher(test_data, model.batch_size, model.num_steps)
        test_loss = 0.0
        test_state = state
        test_memory = memory
        for i, (inputs, targets) in enumerate(test_data_batcher.iterator):
            summaries, loss, step, test_state, test_memory = model.run_eval_step(sess, inputs, targets,
                                                                                 test_state, test_memory, i)
            test_loss = ((test_loss * i) + loss) / (i + 1)

            if supervisor.should_stop():
                supervisor.stop()
        logger.info("Final Test Loss: {}".format(test_loss))
        supervisor.stop()

    else:
        train_data_batcher = Batcher(train_data, model.batch_size, model.num_steps)
        val_data_batcher = Batcher(val_data, model.batch_size, model.num_steps)
        reversed_val_data_batcher = Batcher(val_data, model.batch_size, model.num_steps, reverse=True)
        train_state = state
        val_state = state
        train_memory = memory
        val_memory = memory
        while not supervisor.should_stop() and step < config.max_epoch:

            running_avg_loss, train_state, train_memory = run_step("train",
                                                                   train_data_batcher,
                                                                   model.run_train_step,
                                                                   sess,
                                                                   train_summary_writer,
                                                                   running_avg_loss,
                                                                   step,
                                                                   train_state,
                                                                   train_memory)

            val_running_avg_loss, val_state, val_memory = run_step("validation",
                                                                   val_data_batcher,
                                                                   model.run_eval_step,
                                                                   sess,
                                                                   val_summary_writer,
                                                                   val_running_avg_loss,
                                                                   step,
                                                                   val_state,
                                                                   val_memory)

            # Drop the learning rate.
            model.decay_learning_rate(sess)
            if step % 1000 == 0:
                try:
                    inputs, targets = next(train_data_batcher.iterator)
                except StopIteration:
                    train_data_batcher.refresh_generator()
                    inputs, targets = next(train_data_batcher.iterator)
                image_summary, global_step = model.run_image_summary(sess, inputs, targets, train_state, train_memory)
                train_summary_writer.add_summary(image_summary, global_step)

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
    flags.DEFINE_bool("test", False, "Evaluate model on test data.")
    FLAGS = flags.FLAGS

    from tensorflow.python.platform import flags
    from sys import argv

    flags.FLAGS._parse_flags()
    main(argv)
