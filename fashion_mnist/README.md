# Fashion MNIST Tutorial

In this tutorial, you will learn how to apply Horovod to a [WideResNet](https://arxiv.org/abs/1605.07146) model, trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

## Prerequisites

If this is an in-person session, hosts will set up VM for you and provide you credentials to Jupyter Lab.  If you're working on this tutorial on your own, please follow installation instructions in [INSTALL.md](INSTALL.md).

Let's begin!

## Connect to Jupyter Lab

When you open Jupyter Lab in your browser, you will see a screen similar to this:

![image](https://user-images.githubusercontent.com/16640218/54183442-68aa0480-4461-11e9-872e-89e739ab0937.png)

In this lab, we will use the Terminal and File Editor features.

## Explore model files

On the left hand side, you will see a number of Python files: `fashion_mnist.py`, `fashion_mnist_solution.py`, and a few intermediate files `fashion_mnist_after_step_N.py`.

<img src="https://user-images.githubusercontent.com/16640218/54183508-9000d180-4461-11e9-8fdb-995f065aa4b9.png" width="300"></img>

The first file contains the Keras model that does not have any Horovod code, while the second one has all the Horovod features added.  In this tutorial, we will guide you to transform `fashion_mnist.py` into `fashion_mnist_solution.py` step-by-step.  If you get stuck at any point, you can compare your code with the `fashion_mnist_after_step_N.py` file that corresponds to the step you're at.

Why Keras?  We chose Keras due to its simplicity, and the fact that it will be the way to define models in TensorFlow 2.0.

## Run fashion_mnist.py

Before we go into modifications required to scale our WideResNet model, let's run a single-GPU version of the model.

In the Launcher, click the Terminal button:

<img src="https://user-images.githubusercontent.com/16640218/53534695-d135d080-3ab4-11e9-830b-ea5a9e8581d1.png" width="300"></img>

In the terminal, type:

```
$ cp fashion_mnist.py fashion_mnist_backup.py
$ python fashion_mnist_backup.py --log-dir baseline
```

![image](https://user-images.githubusercontent.com/16640218/53534844-5620ea00-3ab5-11e9-9307-332db459da66.png)

After a few minutes, it will train a few epochs:

![image](https://user-images.githubusercontent.com/16640218/54184767-a4929900-4464-11e9-8a6a-e2fed3f4cd00.png)

Open the browser and load `http://<ip-address-of-vm>:6006/`:

![image](https://user-images.githubusercontent.com/16640218/54184664-69906580-4464-11e9-8a8f-3a0b4028b379.png)

You will see training curves in the TensorBoard.  Let it run.  We will get back to the results later.

## Modify fashion_mnist.py

Double-click `fashion_mnist.py` in the file picker, which will open it in the editor:

![image](https://user-images.githubusercontent.com/16640218/53517877-8c924100-3a84-11e9-9a65-a9054529cc6a.png)

Let's dive into the modifications!

### 1. Add Horovod import

Add the following code after `import tensorflow as tf`:

```python
import horovod.keras as hvd
```

![image](https://user-images.githubusercontent.com/16640218/53517965-c8c5a180-3a84-11e9-9b36-e745bebe84df.png)
(see line 12)

### 2. Initialize Horovod

Add the following code after `args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')`:

```python
# Horovod: initialize Horovod.
hvd.init()
```

![image](https://user-images.githubusercontent.com/16640218/54185178-9d1fbf80-4465-11e9-8617-1f335038a4e0.png)
(see line 36-37)

### 3. Pin GPU to be used by each process

With Horovod, you typically use a single GPU per training process:

<img src="https://user-images.githubusercontent.com/16640218/53518255-7d5fc300-3a85-11e9-8bf3-5d0e8913c14f.png" width="400"></img>

This allows you to greatly simplify the model, since it does not have to deal with the manual placement of tensors.  Instead, you just specify which GPU you'd like to use in the beginning of your script.

Add the following code after `hvd.init()`:

```python
# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
```

![image](https://user-images.githubusercontent.com/16640218/54185222-b9bbf780-4465-11e9-83de-4c587db327ae.png)
(see line 39-43)

### 4. Broadcast the starting epoch from the first worker to everyone else

In `fashion_mnist.py`, we're using the filename of the last checkpoint to determine the epoch to resume training from in case of a failure:

![image](https://user-images.githubusercontent.com/16640218/54185268-d35d3f00-4465-11e9-99eb-96d4b99f1d38.png)

As you scale your workload to multi-node, some of your workers may not have access to the filesystem containing the checkpoint.  For that reason, we make the first worker to determine the epoch to restart from, and *broadcast* that information to the rest of the workers.

To broadcast the starting epoch from the first worker, add the following code:

```python
# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')
```

![image](https://user-images.githubusercontent.com/16640218/53534072-2de3bc00-3ab2-11e9-8cf1-7531542e3202.png)
(see line 52-54)

### 5. Print verbose logs only on the first worker

Horovod uses MPI to run model training workers.  By default, MPI aggregates output from all workers.  To reduce clutter, we recommended that you write logs only on the first worker.

Replace `verbose = 1` with the following code:

```python
# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0
```

![image](https://user-images.githubusercontent.com/16640218/53534314-2244c500-3ab3-11e9-95ef-e7e7b282ab4f.png)
(see line 56-57)

### 6. Read checkpoint only on the first worker

For the same reason as above, we read the checkpoint only on the first worker and *broadcast* the initial state to other workers.

Replace the following code:

```python
# Restore from a previous checkpoint, if initial_epoch is specified.
if resume_from_epoch > 0:
    model = keras.models.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
else:
    ...
```

with:

```python
# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
else:
    ...
```

![image](https://user-images.githubusercontent.com/16640218/53534410-9717ff00-3ab3-11e9-86eb-1bf8299416d2.png)
(see line 91-96)

### 7. Adjust learning rate and add Distributed Optimizer

Horovod uses an operation that averages gradients across workers.  Gradient averaging typically requires a corresponding increase in learning rate to make bigger steps in the direction of a higher-quality gradient.

Replace `opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)` with:

```python
# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                           momentum=args.momentum)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)
```

![image](https://user-images.githubusercontent.com/16640218/53534579-52409800-3ab4-11e9-971e-f7def73c7b36.png)
(see line 116-121)

### 8. Add BroadcastGlobalVariablesCallback

In the previous section, we mentioned that the first worker would broadcast parameters to the rest of the workers.  We will use `horovod.keras.BroadcastGlobalVariablesCallback` to make this happen.

Add `BroadcastGlobalVariablesCallback` as the first element of the `callbacks` list:

```python
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    ...
```

![image](https://user-images.githubusercontent.com/16640218/53535043-1dcddb80-3ab6-11e9-9911-1eb33a1f531c.png)
(see line 139-142)

### 9. Add learning rate warmup

Many models are sensitive to using a large learning rate (LR) immediately after initialization and can benefit from learning rate warmup.  The idea is to start training with lower LR and gradually raise it to a target LR over a few epochs.  Horovod has the convenient `LearningRateWarmupCallback` for the Keras API that implements that logic.

Since we're already using `LearningRateScheduler` in this code, and it modifies learning rate along with `LearningRateWarmupCallback`, there is a possibility of a conflict.  In order to avoid such conflict, we will swap out `LearningRateScheduler` with Horovod `LearningRateScheduleCallback`.

Replace the following code:

```python
def lr_schedule(epoch):
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr


callbacks = [
    ...
    
    keras.callbacks.LearningRateScheduler(lr_schedule),
    ...
```

with:

```python
callbacks = [
    ...

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose),

    # Horovod: after the warmup reduce learning rate by 10 on the 15th, 25th and 35th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=15, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=25, multiplier=1e-1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=25, end_epoch=35, multiplier=1e-2),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=35, multiplier=1e-3),

    ...
```

![image](https://user-images.githubusercontent.com/16640218/53535420-98e3c180-3ab7-11e9-8780-9258081f66c5.png)
(see line 133-142)

Since we've added a new `args.warmup_epochs` argument, we should register it:

```python
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
```

![image](https://user-images.githubusercontent.com/16640218/54185817-284d8500-4467-11e9-9f7d-c6adc5b12cbf.png)
(see line 26-27)

### 10. Save checkpoints & logs only of the first worker

We don't want multiple workers to be overwriting same checkpoint files, since it could lead to corruption.

Replace the following:

```python
callbacks = [
    ...

    keras.callbacks.ModelCheckpoint(args.checkpoint_format),
    keras.callbacks.TensorBoard(args.log_dir)
]
```

with:

```python
callbacks = [
    ...
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))
```

![image](https://user-images.githubusercontent.com/16640218/53535616-4e167980-3ab8-11e9-82d9-82e431dfd621.png)
(see line 145-148)

### 11. Modify training loop to execute fewer steps per epoch

To speed up training, we will execute fewer steps of distributed training.  To keep the total number of examples processed during the training the same, we will do `num_steps / N` steps, where `num_steps` is the original number of steps, and `N` is the total number of workers.

We will also speed up validation by validating `3 * num_validation_steps / N` steps on each worker.  The multiplier **3** provides over-sampling of validation data helps to increase probability that every validation example will be evaluated.

Replace `model.fit_generator(...)` with:

```python
# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data, which helps to increase the probability that every
# validation example will be evaluated.
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=resume_from_epoch,
                    validation_data=test_iter,
                    validation_steps=3 * len(test_iter) // hvd.size())
```

![image](https://user-images.githubusercontent.com/16640218/53536410-283ea400-3abb-11e9-8742-05921b0795de.png)
(see line 152-164)

### 12. Average validation results among workers

Since we're not validating full dataset on each worker anymore, each worker will have different validation results.  To improve validation metric quality and reduce variance, we will average validation results among all workers.

To do so, inject `MetricAverageCallback` after `BroadcastGlobalVariablesCallback`:

```python
callbacks = [
    ...

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    ...
```

![image](https://user-images.githubusercontent.com/16640218/53536553-b2870800-3abb-11e9-88f9-1a2758bd25dd.png)
(see line 135-139)

## Check your work

Congratulations!  If you made it this far, your `fashion_mnist.py` should now be fully distributed.  To verify, you can run the following command in the terminal, which should produce no output:

```
$ diff fashion_mnist.py fashion_mnist_solution.py
$
```

![image](https://user-images.githubusercontent.com/16640218/53536688-23c6bb00-3abc-11e9-9413-1c4ad179653a.png)

## Run distributed fashion_mnist.py

It's time to run your distributed `fashion_mnist.py`.  First, let's check if the single-GPU version completed.  Open the terminal, and verify that it did complete, and interrupt it using Ctrl-C if it did not.

![image](https://user-images.githubusercontent.com/16640218/53536718-448f1080-3abc-11e9-9e22-021dc3ba5de9.png)

Now, run distributed `fashion_mnist.py` using:

```
$ horovodrun -np 4 python fashion_mnist.py --log-dir distributed
```

![image](https://user-images.githubusercontent.com/16640218/53536888-da2aa000-3abc-11e9-9083-43060634433c.png)

After a few minutes, you should see training progress.  It will be faster compared to the single-GPU model:

![image](https://user-images.githubusercontent.com/16640218/53536956-270e7680-3abd-11e9-8f3b-acbe9bbfd085.png)

## Monitor training progress

Open the browser and load `http://<ip-address-of-vm>:6006/`:

![image](https://user-images.githubusercontent.com/16640218/54213792-3ec50200-44a2-11e9-9c7d-fdf9ab1bf94f.png)

By default, TensorBoard shows metric comparison based on the number of epochs.  This is shown on the chart above.  To compare training time it takes to achieve a certain accuracy, select **RELATIVE** in the *Horizontal Axis* selector:

![image](https://user-images.githubusercontent.com/16640218/54213965-94011380-44a2-11e9-9420-138bfe529ec6.png)

### Note

1. Since deep learning training is a stochastic process, you will see variation between accuracy of single-GPU and distributed training runs.  These are normal.

2. You will see approximately **3x** speedup in wall clock time, but not **4x** speedup.  This is expected for this model, since the model is very small and communication overhead plays large role in the training.  As you start training bigger models that take hours, days, or weeks to train, you will generally see better scaling efficiency.

## Parting thoughts

Thanks for following this tutorial!  We're excited to see you apply Horovod to speed up training of your models.
