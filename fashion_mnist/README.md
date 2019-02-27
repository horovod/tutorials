# Fashion MNIST Tutorial

In this tutorial, you will learn how to apply Horovod to a [WideResNet](https://arxiv.org/abs/1605.07146) model, trained on Fashion MNIST dataset.

If this is an in-person session, hosts will set up VM for you and provide you credentials to Jupyter Lab.  If you're working on this tutorial on your own, please follow installation instructions in [INSTALL.md](INSTALL.md).

Let's begin!

## Connect to Jupyter Lab

Once you open Jupyter Lab, you will see screen similar to this:

![image](https://user-images.githubusercontent.com/16640218/53517400-5d2f0480-3a83-11e9-8db9-268037fc6c2f.png)

In this lab, we will use Terminal and File Editor features.

## Explore model files

On the left hand side, you will see two Python files: `fashion_mnist.py` and `fashion_mnist_solution.py`.

<img src="https://user-images.githubusercontent.com/16640218/53517505-9f584600-3a83-11e9-8e1f-9099e9690299.png" width="300"></img>

The first file contains the Keras model that does not have any Horovod code, while the second one has all the Horovod features added.  In this tutorial, we will guide you to transform `fashion_mnist.py` into `fashion_mnist_solution.py` step-by-step.

Why Keras?  We chose Keras due to it's simplicity, and the fact that it will be the way to define models in TensorFlow 2.0.

## Run fashion_mnist.py

Before we go into modifications requires to scale our WideResNet model, let's run a single-GPU version of the model.

In the Launcher, click Terminal button:

<img src="https://user-images.githubusercontent.com/16640218/53534695-d135d080-3ab4-11e9-830b-ea5a9e8581d1.png" width="300"></img>

In the terminal, type:

```
$ cp fashion_mnist.py fashion_mnist_backup.py
$ python fashion_mnist_backup.py --log-dir baseline
```

![image](https://user-images.githubusercontent.com/16640218/53534844-5620ea00-3ab5-11e9-9307-332db459da66.png)

Let it run.  We will get back to the results later.

## Modify fashion_mnist.py

Double-click `fashion_mnist.py` in the file picker, which will open it in the editor:

![image](https://user-images.githubusercontent.com/16640218/53517877-8c924100-3a84-11e9-9a65-a9054529cc6a.png)

Let's dive into the modifications!

### Add Horovod import

Add the following code after `import tensorflow as tf`:

```python
import horovod.keras as hvd
```

![image](https://user-images.githubusercontent.com/16640218/53517965-c8c5a180-3a84-11e9-9b36-e745bebe84df.png)

### Initialize Horovod

Add the following code after `args = parser.parse_args()`:

```python
# Horovod: initialize Horovod.
hvd.init()
```

![image](https://user-images.githubusercontent.com/16640218/53518048-ff9bb780-3a84-11e9-8059-49ea1c5954dc.png)

### Pin GPU to be used by each process

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

![image](https://user-images.githubusercontent.com/16640218/53518149-4689ad00-3a85-11e9-9f59-f22eeba05e73.png)

### Broadcast starting epoch from first worker to everyone else

In `fashion_mnist.py`, we're using filename of the last checkpoint to determine epoch to resume training from in case of a failure:

```python
# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break
```

As you scale your workload to multi-node, some of your workers may not have access to the filesystem containing the checkpoint.  For that reason, we make the first worker to determine epoch to restart from, and *broadcast* that information to the rest of the workers.

This requires adding the following code:

```python
# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')
```

![image](https://user-images.githubusercontent.com/16640218/53534072-2de3bc00-3ab2-11e9-8cf1-7531542e3202.png)

### Print verbose logs only on first worker

Horovod uses MPI to run model training workers.  By default, MPI aggregates output from all workers.  In order to reduce clutter, it's recommended to write logs only on first worker.

Replace `verbose = 1` with the following code:

```python
# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0
```

![image](https://user-images.githubusercontent.com/16640218/53534314-2244c500-3ab3-11e9-95ef-e7e7b282ab4f.png)

### Read checkpoint only on first worker

For the same reason as above, we read checkpoint only the first worker and *broadcast* initial state to other workers.

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

### Adjust learning rate and add Distributed Optimizer

Horovod uses an operation that averages gradients across workers.  Gradient averaging typically requires corresponding increase in learning rate to make bigger steps in the direction of a higher-quality gradient.

Replace `opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)` with:

```python
# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                           momentum=args.momentum)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)
```

![image](https://user-images.githubusercontent.com/16640218/53534579-52409800-3ab4-11e9-971e-f7def73c7b36.png)

### Add BroadcastGlobalVariablesCallback

In previous section, we mentioned that first worker would broadcast parameters to the rest of the workers.  We will use `horovod.keras.BroadcastGlobalVariablesCallback` to make this happen.

Add `BroadcastGlobalVariablesCallback` as the first element of `callbacks` list:

```python
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    ...
```

![image](https://user-images.githubusercontent.com/16640218/53535043-1dcddb80-3ab6-11e9-9911-1eb33a1f531c.png)

### Add learning rate warmup

Many models are sensitive to using large learning rate immediately after initialization and can benefit from learning rate warmup.  The idea is to start training with lower LR and gradually raise it to target LR over few epochs.  Horovod has convenient `LearningRateWarmupCallback` for Keras API that implements that logic.

Since we're already using `LearningRateScheduler` in this code, and it modifies learning rate along with `LearningRateWarmupCallback`, there is a possibility of a conflict.  In order to avoid such conflict, we will swap out `LearningRateScheduler` with Horovod `LearningRateScheduleCallback`.

We will replace the following code:

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

Since we've added new `args.warmup_epochs` argument, we should register it:

```python
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
```

![image](https://user-images.githubusercontent.com/16640218/53535703-9d5caa00-3ab8-11e9-85aa-0bc6f93f9826.png)

### Save checkpoints & logs only of the first worker

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

### Modify training loop to execute fewer steps per epoch

In order to speed up training, we will execute fewer steps of distributed training.  In order to keep the total number of examples processed during the training the same, we will do `num_steps / N` steps, where `num_steps` is the original number of steps, and `N` is the total number of workers.

We will also speed-up validation by validating `3 * num_validation_steps / N` steps on each worker.  Multiplier **3** provides over-sampling of validation data helps to increase probability that every validation example will be evaluated.

Replace `model.fit_generator(...)` with:

```python
# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data helps to increase probability that every validation
# example will be evaluated.
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

### Average validation results among workers

Since we're not validating full dataset on each worker anymore, each worker will have different validation results.  To improve validation metric quality and reduce variance, we will average validation results among all workers.

To do so, we will inject `MetricAverageCallback` after `BroadcastGlobalVariablesCallback`:

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

### Check your work

Congratulations!  If you made it this far, your `fashion_mnist.py` should now be fully distributed.  In order to verify, you can run the following command in the terminal, which should produce no output:

```
$ diff fashion_mnist.py fashion_mnist_solution.py
$
```

![image](https://user-images.githubusercontent.com/16640218/53536688-23c6bb00-3abc-11e9-9413-1c4ad179653a.png)

## Run distributed fashion_mnist.py

It's time to run your distributed `fashion_mnist.py`.  First, let's check if the single-GPU version completed.  Open the terminal, and verify that it did complete, and interrupt it using Ctrl-C if it did not.

![image](https://user-images.githubusercontent.com/16640218/53536718-448f1080-3abc-11e9-9e22-021dc3ba5de9.png)

You should now remove checkpoint files, since we will not need them:

```
$ rm checkpoint*
$ ls checkpoint*
```

![image](https://user-images.githubusercontent.com/16640218/53536860-c1ba8580-3abc-11e9-9276-46fcb6c53d4c.png)

Now, run distributed `fashion_mnist.py` using:

```
$ horovodrun -np 4 python fashion_mnist.py --log-dir distributed
```

![image](https://user-images.githubusercontent.com/16640218/53536888-da2aa000-3abc-11e9-9083-43060634433c.png)

After a few minutes, you should see training progress.  It will be faster compared to single-GPU model:

![image](https://user-images.githubusercontent.com/16640218/53536956-270e7680-3abd-11e9-8f3b-acbe9bbfd085.png)

## Monitor training progress

In order to monitor training progress and compare runs, we will use TensorBoard.

Open another Terminal in Launcher, and exectute:

```
$ tensorboard --logdir .
```

![image](https://user-images.githubusercontent.com/16640218/53537203-072b8280-3abe-11e9-8ffa-3cb1b1ed5922.png)

Now, open the browser and load `http://<ip-address-of-vm>:6006/`:

![image](https://user-images.githubusercontent.com/16640218/53537263-35a95d80-3abe-11e9-9f88-45c15a4eab11.png)

You will now be able to compare accuracy and training time of both models.

### Note

1. Since deep learning training is a stochastic process, you will see variation between accuracy of single-GPU and distributed training runs.  These are normal.

2. You will see approximately **3x** speedup in wall clock time, but not **4x** speedup.  This is expected for this model, since the model is very small and communication overhead plays large role in the training.  As you start training bigger models that take hours, days, or weeks to train, you will generally see better scaling efficiency.

## Parting thoughts

Thanks for following this tutorial!  We're excited to see you apply Horovod to speed up training of your models.