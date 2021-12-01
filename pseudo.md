# Arbsr:

dataloader, model, loss checkpoint, trainer
loop trainer() until it calls terminate().
On terminate(), test() and finish, or step epoch and return bool epoch finished tracker. 
Finalise by close logger.

## train:
step scheduler, step loss, iter epoch counter.
start loss log,
set model to training mode.

first epoch: use int scale factors.
### custom dataloader fn:
```# train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
if dataset.first_epoch and len(scale) > 1 and dataset.train:
    idx_integer_scale_list = [9, 19, 29]
    rand_idx = random.randrange(0, len(idx_integer_scale_list))
    dataset.set_scale(idx_integer_scale_list[rand_idx])
    
if not dataset.first_epoch and len(scale) > 1 and dataset.train:
    # train on all scale factors for remaining epochs
    idx_scale = random.randrange(0, len(scale))
    dataset.set_scale(idx_scale)
```

idx_scale = [9,19,29] -- ??? Why? Dataset organisation?

## `Trainer().train loop iterations` 
called once per epoch in main while loop, terminates according to epoch counter.
1. get data on device, set precision ("prepare")
1. determine scale
1. zero grad
1. pass scale to model
1. apply model(lr)
1. calc loss(sr, hr)
1. loss backward
1. step optimiser
1. save model

## `Trainer.test()`
1. set model eval
1. with no_grad()
1. loop scales
1. set _val dataloader_ scale to loop value
1. enumerate val_dataloader
1. prepare data
1. set _model_ scale to loop value
1. apply model to val data
1. append image result to list **could do same based on preview indices**

**n.b.** - ARBSR crops border before validation and previews, but includes border in loss function. Checks for smallest scale integer factor in [1,2,5,10,20,50] and uses it to determine new size...


# ESRGAN+ trainer
0. set up loggers, dirs, load checkpoint if, cfg, etc.
1. create dloaders according to opts
1. determine epoch and iteration total counts
1. create model according to opts, resume if necessary, else set curr_iter/epoch to 0
## epoch loop in range start, total
### 4. train iteration loop in enumerate train_dataloader
1. step iter count, break if > total
1. update lr
1. data and optimize in methods
1. log
1. check step vs val_steps if:
### 5. validation loop in enumerate val_dataloader
1. "prepare" data to device
1. set model eval, no grad
1. apply model(lr), cleanup set model train
1. get visuals, log
1. calc psnr, average, log
1. check checkpoint step/freq.  
## 6. cleanup
1. save model, log



