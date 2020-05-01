import tensorflow as tf
from utils import dataset_split, dataset_split_with_test


class RegressionExperiment:
    def __init__(self, tree_model, simple_model, config, tree_root=None, leaves=None,
                 run_preprocessor=False, prepressor_func=None, use_test=True, baselines=False, expanded_x=None):
        if run_preprocessor is False or prepressor_func is None:
            assert tree_root is not None and leaves is not None, 'Cannot skip preprocessor without providing ' \
                                                                 'training data'
            self.tree_root = tree_root
            self.leaves = leaves
        else:
            self.tree_root, self.leaves = prepressor_func(config['data_file'])
        # these two should be an already instantiated model instance
        self.tree_model = tree_model
        self.simple_model = simple_model
        self.config = config
        if use_test:
            self.all_x, self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y, \
                self.train_idx, self.valid_idx, self.test_idx = dataset_split_with_test(self.leaves, self.config['seed'])
        else:
            self.all_x, self.train_x, self.valid_x, self.train_y, self.valid_y, self.train_idx, self.valid_idx = \
                dataset_split(self.leaves, self.config['seed'])
        self.loss_function = tf.losses.mean_squared_error
        self.calc_auc = False
        self.dendro_auc = None
        self.simple_auc = None
        self.baselines = baselines
        if self.baselines:
            self.parsimony_x = tf.reshape(tf.identity([[1.0] for _ in range(len(self.leaves))]), shape=(len(self.leaves), 1, 1))
            self.expanded_x = expanded_x  # todo: add the one-hot placement features

    @tf.function
    def tree_train_step(self, optimizer, dendro_reg=True):
        with tf.GradientTape() as tape:
            """
            gathering predictions if we are running the parsimony style baseline
            """
            if self.baselines:
                y_hat, delta = self.tree_model(tf.squeeze(self.parsimony_x, axis=1))
            else:
                if self.config['l2']:
                    y_hat, delta, l2 = self.tree_model(self.all_x)
                else:
                    y_hat, delta = self.tree_model(self.all_x)
            y_hat_train = tf.gather(y_hat, self.train_idx)
            y_hat_valid = tf.gather(y_hat, self.valid_idx)
            dendronet_loss = self.config['delta_penalty_factor'] * delta
            predict_loss = self.loss_function(self.train_y, y_hat_train) * self.config['loss_scale']
            if dendro_reg:
                loss = predict_loss + dendronet_loss
            else:
                loss = predict_loss
            if self.config['l2']:
                l2_loss = l2 * self.config['l2_penalty_factor'] # todo: make the l2 penalty factor scale with the number of nodes
                loss = loss + l2_loss
            valid_loss = self.loss_function(self.valid_y, y_hat_valid)
        gradients = tape.gradient(loss, self.tree_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.tree_model.trainable_variables))
        if self.calc_auc:
            self.dendro_auc.update_state(self.valid_y[:,0:1], y_hat_valid[:,0:1])
        return tf.reduce_mean(predict_loss), dendronet_loss, tf.reduce_mean(valid_loss)

    # todo: why does making this a tf function cause it to get mixed up with the other train function at run time??
    @tf.function
    def simple_train_step(self, optimizer):
        with tf.GradientTape() as tape_simple:
            # y_hat = self.simple_model.call(self.train_x)
            y_hat = tf.squeeze(tf.map_fn(lambda x_i: tf.squeeze(self.simple_model.call(x_i), axis=1), self.train_x),
                               axis=1)
            predict_loss = self.loss_function(self.train_y, y_hat)
        gradients = tape_simple.gradient(predict_loss, self.simple_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.simple_model.trainable_variables))
        return tf.reduce_mean(predict_loss)

    @tf.function
    def simple_valid_step(self):
        # y_hat = self.simple_model.call(self.valid_x)
        y_hat = tf.squeeze(tf.map_fn(lambda x_i: tf.squeeze(self.simple_model.call(x_i), axis=1), self.valid_x), axis=1)
        validation_predict_loss = self.loss_function(self.valid_y, y_hat)
        if self.calc_auc:
            self.simple_auc.update_state(self.valid_y[:,0:1], y_hat[:,0:1])
        return tf.reduce_mean(validation_predict_loss)

    def train_dendronet(self, run_test=False, start_dendroreg=0):
        dendronet_losses = list()
        prediction_losses = list()
        validation_losses = list()
        validation_aucs = list()
        optimizer = tf.optimizers.Adam(learning_rate=self.config['lr'], amsgrad=True)
        # optimizer = tf.optimizers.SGD(learning_rate=self.config['lr'], decay=0.01, momentum=0.01)
        dendro_reg=False
        for step in range(self.config['num_steps']):
            if step==start_dendroreg:
                dendro_reg=True
            if self.dendro_auc is not None:
                self.dendro_auc.reset_states()
            prediction_loss, dendronet_loss, validation_loss = self.tree_train_step(optimizer, dendro_reg)
            if step % self.config['validation_interval'] == 0:
                print('step ' + str(step))
                print('Total loss: ' + str(prediction_loss + dendronet_loss))
                print('Dendronet loss: ' + str(dendronet_loss))
                print('Train prediction loss: ' + str(prediction_loss))
                print('Validation prediction loss: ' + str(validation_loss))
                print('')
                print('')
                dendronet_losses.append(dendronet_loss.numpy()[0])
                prediction_losses.append(prediction_loss.numpy())
                validation_losses.append(validation_loss.numpy())
                if self.calc_auc:
                    validation_aucs.append(self.dendro_auc.result().numpy())
            if step + 1 == self.config['num_steps']:
                print('Done training dendronet')

        return dendronet_losses, prediction_losses, validation_losses, validation_aucs

    def train_simple_model(self, get_activations=False, run_test=False):
        prediction_losses = list()
        validation_losses = list()
        validation_aucs = list()
        activations = list()
        # optimizer = tf.optimizers.Adam(learning_rate=self.config['lr'])
        optimizer = tf.optimizers.Adam(learning_rate=self.config['lr'], amsgrad=True)
        # optimizer = tf.optimizers.SGD(learning_rate=self.config['lr'], decay=0.01, momentum=0.01)
        for step in range(self.config['num_steps']):
            prediction_loss = self.simple_train_step(optimizer)
            if step % self.config['validation_interval'] == 0:
                validation_loss = self.simple_valid_step()
                print('step ' + str(step))
                print('Train prediction loss: ' + str(prediction_loss))
                print('Validation prediction loss: ' + str(validation_loss))
                print('')
                print('')
                prediction_losses.append(prediction_loss.numpy())
                validation_losses.append(validation_loss.numpy())
                if self.calc_auc:
                    validation_aucs.append(self.simple_auc.result().numpy())
                    self.simple_auc.reset_states()
            if step + 1 == self.config['num_steps']:
                print('Done training simple model')
        if get_activations:
            for input in self.train_x.numpy():
                activations.append(self.simple_model.get_activations(input))
            return prediction_losses, validation_losses, validation_aucs, activations
        return prediction_losses, validation_losses, validation_aucs


class ClassificationExperiment(RegressionExperiment):

    def __init__(self, tree_model, simple_model, config, tree_root=None, leaves=None,
                 run_preprocessor=False, prepressor_func=None, use_test=False, baselines=False, expanded_x=None):
        super(ClassificationExperiment, self).__init__(tree_model, simple_model, config, tree_root, leaves,
                                                       run_preprocessor, prepressor_func, use_test, baselines, expanded_x)
        self.loss_function = tf.losses.categorical_crossentropy
        self.train_y = tf.keras.utils.to_categorical(self.train_y, num_classes=self.config['num_classes'])
        self.valid_y = tf.keras.utils.to_categorical(self.valid_y, num_classes=self.config['num_classes'])
        if use_test:
            self.test_y = tf.keras.utils.to_categorical(self.test_y, num_classes=self.config['num_classes'])
        self.calc_auc = True
        self.dendro_auc = tf.metrics.AUC(name='dendro_auc')
        self.simple_auc = tf.metrics.AUC(name='simple_auc')


class LogRegExperiment(ClassificationExperiment):
    def __init__(self, tree_model, simple_model, config, tree_root=None, leaves=None,
                 run_preprocessor=False, prepressor_func=None, use_test=False, baselines=False, expanded_x=None):
        super(LogRegExperiment, self).__init__(tree_model, simple_model, config, tree_root, leaves,
                                                run_preprocessor, prepressor_func, use_test, baselines, expanded_x)

    # todo: why does making this a tf function cause it to get mixed up with the other train function at run time??
    @tf.function
    def simple_train_step(self, optimizer):
        with tf.GradientTape() as tape:
            if self.baselines:
                # train_x = tf.gather(self.expanded_x, self.train_idx)
                train_x = tf.cast(tf.gather(self.expanded_x, self.train_idx), dtype=tf.float32)
                y_hat = tf.squeeze(tf.concat(tf.map_fn(
                    lambda x_i: self.simple_model.call(
                        tf.transpose(tf.expand_dims(x_i, 0))), train_x), axis=0), axis=1)
            #  todo: figure out if increasing parallel iterations can be safely done
            else:
                y_hat = tf.squeeze(tf.concat(tf.map_fn(
                    lambda x_i: self.simple_model.call(
                        tf.transpose(tf.expand_dims(x_i, 0))), self.train_x), axis=0), axis=1)
            # # todo: adding this on January 14th, make sure does not break other files
            # y_hat = tf.squeeze(tf.concat(tf.map_fn(
            #     lambda x_i: self.simple_model.call(
            #         tf.transpose(tf.expand_dims(x_i, 0))), self.train_x), axis=0), axis=-1)
            # y_hat = tf.squeeze(tf.map_fn(lambda x_i: tf.squeeze(self.simple_model.call(x_i), axis=1), self.train_x),
            #                    axis=1)
            # y_hat = tf.squeeze(tf.map_fn(lambda x_i: tf.squeeze(self.simple_model.call(x_i), axis=1), self.train_x),
            #                    axis=1)
            predict_loss = self.loss_function(self.train_y, y_hat)
        gradients = tape.gradient(predict_loss, self.simple_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.simple_model.trainable_variables))
        return tf.reduce_mean(predict_loss)

    @tf.function
    def simple_valid_step(self):
        if self.baselines:
            valid_x = tf.cast(tf.gather(self.expanded_x, self.valid_idx), dtype=tf.float32)
            y_hat = tf.squeeze(tf.concat(tf.map_fn(
                lambda x_i: self.simple_model.call(
                    tf.transpose(tf.expand_dims(x_i, 0))), valid_x), axis=0), axis=1)
        else:
            y_hat = tf.squeeze(tf.concat(tf.map_fn(
                lambda x_i: self.simple_model.call(
                    tf.transpose(tf.expand_dims(x_i, 0))), self.valid_x), axis=0), axis=1)
        # # todo: as above - adding this on January 14th, make sure does not break other files
        # y_hat = tf.squeeze(tf.concat(tf.map_fn(
        #     lambda x_i: self.simple_model.call(
        #         tf.transpose(tf.expand_dims(x_i, 0))), self.valid_x), axis=0), axis=-1)

        validation_predict_loss = self.loss_function(self.valid_y, y_hat)
        if self.calc_auc:
            self.simple_auc.update_state(self.valid_y[:,0:1], y_hat[:,0:1])
        return tf.reduce_mean(validation_predict_loss)

    def retrieve_predictions(self):
        """
        :return: a list of predictions and targets for the two models, allowing for plotting of an ROC curve
        """
        targets = self.valid_y
        if self.baselines:
            dendro_predictions, _ = self.tree_model(self.parsimony_x)
            dendro_predictions = tf.gather(dendro_predictions, self.valid_idx)
            simple_x = tf.gather(self.expanded_x, self.valid_idx)
            simple_predictions = tf.squeeze(tf.concat(tf.map_fn(
                lambda x_i: self.simple_model.call(
                    tf.transpose(tf.expand_dims(x_i, 0))), simple_x), axis=0), axis=1)
        else:
            simple_predictions = tf.squeeze(tf.concat(tf.map_fn(
                lambda x_i: self.simple_model.call(
                    tf.transpose(tf.expand_dims(x_i, 0))), self.valid_x), axis=0), axis=1)
            dendro_predictions, _ = self.tree_model(self.all_x)
            dendro_predictions = tf.gather(dendro_predictions, self.valid_idx)
        return targets, dendro_predictions, simple_predictions