import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow, flow_error, warp_image
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE
from tqdm import tqdm
import time
slim = tf.contrib.slim


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False, global_step=None):
        # self.global_step = slim.get_or_create_global_step()
        # with tf.variable_scope("Global", reuse=True) as scope:
        #     self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.int64, trainable=False)
        # print(self.global_step)
        if global_step is not None:
            self.global_step = global_step
        else:
            self.global_step = slim.get_or_create_global_step()
        print(self.global_step)

        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):
        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        # Convert from RGB -> BGR
        input_a = input_a[..., [2, 1, 0]]
        input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

        inputs = {
            'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
            'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        }
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            pred_flow = sess.run(pred_flow)[0, :, :, :]

            unique_name = 'flow-' + str(uuid.uuid4())
            if save_image:
                flow_img = flow_to_image(pred_flow)
                full_out_path = os.path.join(out_path, unique_name + '.png')
                imsave(full_out_path, flow_img)

            if save_flo:
                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow, full_out_path)

    def validate(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints):
        """
        This code can be used to test the performance of trained model.,,,
        :param log_dir:
        :param training_schedule:
        :param input_a:
        :param input_b:
        :param flow:
        :param checkpoints:
        :return:
        """

        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        # print(flow.shape())

        # Test input data by warping image b to see if the result is the same as image_a, and log the msr error.
        gt_r_image_a = tf.py_func(warp_image, [input_b, flow], tf.uint8)
        tf.summary.image("gt_r_image_a", gt_r_image_a, max_outputs=2)

        predictions = self.model(inputs, training_schedule)

        epe = tf.py_func(flow_error, [flow, predictions], tf.float32)
        tf.summary.scalar("EPE_over_reasonable_pixels", epe)
        # total_loss, _ = self.loss(flow, predictions)
        #
        # tf.summary.scalar('loss', total_loss)
        # tf.summary.scalar('average_epe', average_epe)

        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

            true_flow_0 = flow[0, :, :, :]
            true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
            true_flow_1 = flow[1, :, :, :]
            true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
            true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
            tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        merged = tf.summary.merge_all()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # sess.run(init_op)
            saver.restore(sess, checkpoints)
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            val_writer = tf.summary.FileWriter(log_dir + '/val_summary', sess.graph)

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            self.global_step = self.global_step+1

            epe = []
            t_l = []
            try:
                for i in tqdm(range(80)):

                    summary, a_total_loss, a_epe = sess.run([merged, total_loss, average_epe])
                    val_writer.add_summary(summary, i)
                    epe.append(a_epe)
                    t_l.append(a_total_loss)
                    # time.sleep(3)
            except Exception, e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

        print('Average total loss is', sum(t_l)/80.)
        print('Validation average end points error is', sum(epe)/80.)


    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss, average_epe = self.loss(flow, predictions)
        tf.assert_rank(average_epe, 0)

        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('average_epe', average_epe)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)

                # for var in variables_to_restore[:20]:
                #     print (var.op.name)
                #     print (var.op.name.split(new_scope + '/')[1])

                # renamed_variables = variables_to_restore
                # if scope == 'FlowNet2' or 'FlowNetS':
                if scope == new_scope:
                    renamed_variables = variables_to_restore
                else:
                    renamed_variables = {
                        var.op.name.split(new_scope + '/')[1]: var
                        for var in variables_to_restore
                    }

                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            slim.learning.train(
                train_op,
                log_dir,
                session_config=config,
                global_step=self.global_step,
                save_summaries_secs=12,
                save_interval_secs=60,
                number_of_steps=training_schedule['max_iter']
            )

    def restore_train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None,
                      set_global_step=False):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }

        predictions = self.model(inputs, training_schedule)
        total_loss, average_epe = self.loss(flow, predictions)
        # tf.assert_rank(average_epe, 0)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # with tf.Session() as sess:
        #     print("Global step @1 is", sess.run(self.global_step))
        #     print("Learning rate is: ", sess.run(self.learning_rate))

        if set_global_step:
            variables_to_restore = slim.get_variables_to_restore(exclude=['global_step'])
        else:
            variables_to_restore = slim.get_variables_to_restore()
        print('#########################')
        # variables_to_restore = slim.get_variables()
        for v in variables_to_restore:
            print(v)

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        print('#########################')

        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v)

        if checkpoints is None:
            raise Exception('Checkpoint file not given.')

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoints, variables_to_restore)
        print('#########################')
        print(init_assign_op)
        print('#########################')

        for k,v in init_feed_dict.items():
            if k.name =='placeholder/global_step:0':
                print (k,v)
        print('#########################')
        # print(init_feed_dict['placeholder/global_step:0'])

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            slim.learning.train(
                train_op,
                log_dir,
                init_fn=InitAssignFn,
                session_config=config,
                global_step=self.global_step,
                save_summaries_secs=12,
                save_interval_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
