import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_structures.gen_tree import generate_random_tree
from models.custom_models import TreeModelLogReg, LogRegModel
from utils import train_valid_split_indices

num_steps = 500
weights_dim = 4
output_dim = 2
layer_shape = (weights_dim, output_dim)
mutation_rate = 0.1
root, leaves = generate_random_tree(depth=5, weights_dim=weights_dim, mutation_rate=mutation_rate)
train_indices, validation_indices = train_valid_split_indices(max_index=len(leaves))
train_idx = tf.identity(train_indices)
validation_idx = tf.identity(validation_indices)
delta_penalty_factor = 0.1

tree_model = TreeModelLogReg(root, leaves, layer_shape)


y = list()
y_valid = list()
leaf_x = list()
for i in range(len(leaves)):
    leaf = leaves[i]
    leaf_x.append([leaf.x ** i for i in range(weights_dim)])  # first term is the bias / intercept term
    if i in train_indices:
        y.append(leaf.y)
    else:
        y_valid.append(leaf.y)

# for dummy data: transforming y values into classification
mean_y = np.mean(y)

for i in range(len(y)):
    if y[i] > mean_y:
        y[i] = 1.0
    else:
        y[i] = 0.0
for i in range(len(y_valid)):
    if y_valid[i] > mean_y:
        y_valid[i] = 1.0
    else:
        y_valid[i] = 0.0

leaf_x = tf.identity(leaf_x)
optimizer = tf.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
classification_loss = tf.keras.metrics.Mean(name='classification_loss')
classification_accuracy = tf.metrics.Accuracy(name='classification_accuracy')
regularization_loss = tf.keras.metrics.Mean(name='regularization_loss')
validation_loss = tf.keras.metrics.Mean(name='validation_loss')
validation_accuracy = tf.metrics.Accuracy(name='validation_accuracy')


# @tf.function
def train_step(x, y_t, y_v):
    y_t = tf.keras.utils.to_categorical(y_t, num_classes=output_dim)
    y_v = tf.keras.utils.to_categorical(y_v, num_classes=output_dim)
    with tf.GradientTape() as tape:
        y_hat, delta = tree_model(x)
        # separate train examples from validation examples
        y_hat_train = tf.gather(y_hat, train_idx)
        y_hat_valid = tf.gather(y_hat, validation_idx)
        mutation_loss = delta_penalty_factor * delta
        class_loss = tf.losses.categorical_crossentropy(y_t, y_hat_train)
        loss = class_loss + mutation_loss
        valid_loss = tf.losses.categorical_crossentropy(y_v, y_hat_valid)  # note this does not get passed to optimizer
    gradients = tape.gradient(loss, tree_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tree_model.trainable_variables))

    train_loss(tf.reduce_sum(loss)) # fudgy, sum of all losses
    classification_loss(tf.reduce_mean(class_loss))
    regularization_loss(tf.reduce_sum(mutation_loss))
    validation_loss(tf.reduce_mean(valid_loss))
    classification_accuracy.update_state(tf.argmax(y_t, 1), tf.argmax(y_hat_train, 1))
    validation_accuracy.update_state(tf.argmax(y_v, 1), tf.argmax(y_hat_valid, 1))


# store the losses for plotting
mutation_losses = list()
train_losses = list()
validation_losses = list()
# tree_model.build([weights_dim])


for step in range(num_steps):
    train_step(leaf_x, y, y_valid)
    # if step == 1:
        # print(tree_model.summary())
    if step % 100 == 0:
        print('step ' + str(step))
        print('Loss: ' + str(train_loss.result().numpy()))
        mut_loss = regularization_loss.result().numpy()
        mutation_losses.append(mut_loss)
        print('Mutation loss: ' + str(mut_loss))
        trn_loss = classification_loss.result().numpy()
        train_losses.append(trn_loss)
        print('Train classification loss: ' + str(trn_loss))
        vld_loss = validation_loss.result().numpy()
        validation_losses.append(vld_loss)
        print('Validation classification loss: ' + str(vld_loss))
        class_accuracy = classification_accuracy.result().numpy()
        valid_accuracy = validation_accuracy.result().numpy()
        print('Train accuracy: ' + str(class_accuracy))
        print('Validation accuracy: ' + str(valid_accuracy))

        print('')

# plot and save results
plt.plot(train_losses, label='train')
plt.plot(validation_losses, label='validation')
plt.xlabel('100\'s of steps')
plt.ylabel('loss')
plt.legend(loc='best')
plt.grid()
plt.savefig('classification_loss.png')
plt.clf()

plt.plot(mutation_losses, label='mutation')
plt.xlabel('100\'s of steps')
plt.ylabel('loss')
plt.grid()
plt.savefig('mutation.png')
plt.clf()

"""
Begin simple logistic regression block
"""


# now we want to make a comparison to straight linear regression with no weight mutation
print('Tree network complete - starting linear regression training')
lin_model = LogRegModel(layer_shape=layer_shape)
input_shape = (4, 1)
lin_model.build(input_shape)
# we need to separate the train leaves from the validation leaves, use the same split as the tree model
linreg_train_x = tf.gather(leaf_x, train_idx)
linreg_valid_x = tf.gather(leaf_x, validation_idx)
linreg_optimizer = tf.optimizers.Adam()

linreg_train_loss = tf.keras.metrics.Mean(name='linreg_train_loss')
linreg_valid_loss = tf.keras.metrics.Mean(name='linreg_valid_loss')


# @tf.function
def linreg_train_step(x, y):
    with tf.GradientTape() as tape:
        y = tf.keras.utils.to_categorical(y, num_classes=output_dim)
        # y_hat = lin_model.call(x)
        # y_hat = tf.concat([lin_model.call(tf.expand_dims(x_i, 1)) for x_i in x], axis=0)
        # y_hat = tf.map_fn(lambda x_i: lin_model.call(tf.expand_dims(x_i, 0)), x)
        y_hat = tf.squeeze(tf.concat(tf.map_fn(lambda x_i: lin_model.call(tf.transpose(tf.expand_dims(x_i, 0))), x), axis=0),
                   axis=1)
        # loss = tf.losses.mean_squared_error(tf.expand_dims(y, 0), y_hat)
        class_loss = tf.losses.categorical_crossentropy(y, y_hat)
    gradients = tape.gradient(class_loss, lin_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lin_model.trainable_variables))
    linreg_train_loss(tf.reduce_mean(class_loss))


@tf.function
def linreg_valid_step(x, y):
    y = tf.keras.utils.to_categorical(y, num_classes=output_dim)
    # y_hat = lin_model.call(x)
    # y_hat = tf.concat([lin_model.call(tf.expand_dims(x_i, 1)) for x_i in x], axis=0)
    y_hat = tf.squeeze(tf.concat(tf.map_fn(lambda x_i: lin_model.call(tf.transpose(tf.expand_dims(x_i, 0))), x), axis=0),
               axis=1)
    # loss = tf.losses.mean_squared_error(tf.expand_dims(y, 0), y_hat)
    class_loss = tf.losses.categorical_crossentropy(y, y_hat)
    linreg_valid_loss(tf.reduce_mean(class_loss))


# gather losses for plotting
train_losses = list()
validation_losses = list()

# run training
for step in range(num_steps):
    linreg_train_step(linreg_train_x, y)
    # if step == 1:
    #     print(lin_model.summary())
    if step % 100 == 0:
        print('step ' + str(step))
        trn_loss = linreg_train_loss.result().numpy()
        print('Train Loss: ' + str(trn_loss))
        train_losses.append(trn_loss)
        linreg_valid_step(linreg_valid_x, y_valid)
        vld_loss = linreg_valid_loss.result().numpy()
        validation_losses.append(vld_loss)
        print('Validation loss: ' + str(vld_loss))
        print(''
              ''
              '')

# plot and save results
plt.plot(train_losses, label='train')
plt.plot(validation_losses, label='validation')
plt.xlabel('100\'s of steps')
plt.ylabel('loss')
plt.legend(loc='best')
plt.grid()
plt.savefig('linreg_loss.png')
plt.clf()


"""
End simple logistic regression block
"""