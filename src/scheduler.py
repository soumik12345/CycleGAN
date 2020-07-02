import tensorflow as tf


class LinearDecay(tf.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(
            initial_value=initial_learning_rate,
            trainable=False, dtype=tf.float32
        )

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (
                    1 - 1 / (self._steps - self._step_decay
                             ) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self._initial_learning_rate,
            'total_steps': self._steps,
            'step_decay': self._step_decay,
        }
