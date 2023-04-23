import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cifar import obtain_cifar10_dataset
from utils import AverageMeter, ModelEMA, accuracy, save_checkpoint
import wideresnet as models
# from dataset.cifar import CIFAR10Preprocessor



class CosineWarmupScheduler:
    def __init__(self, optimizer, epochs, eval_step, num_cycles=7./16., last_epoch=-1):
        """
        Description: Implements the cosine learning rate schedule with warmup for a PyTorch optimizer
        Inputs:
            - optimizer: PyTorch optimizer object to apply the learning rate schedule to
            - epochs: number of epochs to train for
            - eval_step: number of batches between each evaluation of the validation set
            - num_cycles: number of cosine cycles to complete over the course of training (default is 7/16)
            - last_epoch: index of the last epoch completed (default is -1)
        Outputs: None
        """
        self.total_steps = epochs * eval_step
        self.optimizer = optimizer
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch

    def _lr_lambda(self, current_step):
        """
        Description: Calculates the learning rate scaling factor for a given step using a cosine function with linear warmup
        Inputs:
            - current_step: index of the current step in the training process
        Outputs:
            - lr_scale: scaling factor to apply to the learning rate at the current step
        """
        # Calculate the proportion of the total steps completed so far
        no_progress = float(current_step) / float(self.total_steps)
        # Calculate the scaling factor for the current step using a cosine function with linear warmup
        lr_scale = max(0., math.cos(math.pi * self.num_cycles * no_progress))
        # Return the scaling factor
        return lr_scale

    def get_scheduler(self):
        """
        Description: Returns a PyTorch learning rate scheduler object that implements the cosine learning rate schedule with warmup
        Inputs: None
        Outputs:
            - scheduler: PyTorch learning rate scheduler object
        """
        # Create a PyTorch LambdaLR scheduler object using the _lr_lambda method
        scheduler = LambdaLR(self.optimizer, self._lr_lambda, self.last_epoch)
        # Return the scheduler object
        return scheduler


def filter_decay_params(model, no_decay):
    """
    Description: Separates model parameters into decay and no_decay groups based on whether their names contain specified keywords
    Inputs:
        - model: PyTorch model object to filter parameters for
        - no_decay: list of keyword strings to identify parameters that should not decay
    Outputs:
        - decay_parameters: list of PyTorch parameter objects that should decay
        - no_decay_parameters: list of PyTorch parameter objects that should not decay
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, param in model.named_parameters():
        # Check if any of the no_decay keywords are in the parameter name
        if any(nd in name for nd in no_decay):
            # If so, add the parameter to the no_decay group
            no_decay_parameters.append(param)
        else:
            # Otherwise, add the parameter to the decay group
            decay_parameters.append(param)

    return decay_parameters, no_decay_parameters


BATCH_SIZE = 64   
EVAL_STEPS = 1024
LEARNING_RATE = 0.03

NUM_EPOCHS = 200
LAMBDA_U = 1
WEIGHT_DECAY = 5e-4
MU = 7
TEMPERATURE = 1
THRESHOLD = 0.95
RESULT_PATH = 'blop'

best_accuracy = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """
    Description: The main function that runs the FixMatch algorithm on the CIFAR-10 dataset
    """

    global best_accuracy  # Allows for global access to the best accuracy during training

    # Create result directory and initialize TensorBoard
    os.makedirs(RESULT_PATH, exist_ok=True)
    tensor_board = SummaryWriter(RESULT_PATH)

    # Load CIFAR-10 dataset
    labeled_dataset, unlabeled_dataset, test_dataset = obtain_cifar10_dataset('./data')

    # Initialize data loaders, RandomSampler pour toute nos données d'entrainement
    labeled_data_loader = DataLoader(labeled_dataset, sampler=RandomSampler(labeled_dataset),
                                      batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
    # FOr one label image we take 7 images unlabel => BATCH_SIZE*MU
    unlabeled_data_loader = DataLoader(unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
                                        batch_size=BATCH_SIZE*MU, num_workers=4, drop_last=True) 
    test_data_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                    batch_size=BATCH_SIZE, num_workers=4, drop_last=True)


    # Initialize model
    model = models.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=10)
    model.to(device)

    # FIXMATCH PAPER
    # In all of our models and experiments, we use simple weight decay regularization => 0.0005
    # We also found that using the Adam optimizer resulted in worse performance 
    #and instead use standard SGD with momentum. We did not find a substantial difference 
    # between standard and Nesterov momentum
    no_decay = ['bias', 'bn']
    decay_params, no_decay_params = filter_decay_params(model, no_decay)
    optimizer = optim.SGD([{'params': decay_params, 'weight_decay': WEIGHT_DECAY},
                            {'params': no_decay_params, 'weight_decay': 0.0}], lr=LEARNING_RATE,
                            momentum=0.9, nesterov=True)

    # FIXMATCH PAPER
    # For a learning rate schedule, we use a cosine learning rate decay
    cosine_warmup_scheduler = CosineWarmupScheduler(optimizer, NUM_EPOCHS, EVAL_STEPS)
    scheduler = cosine_warmup_scheduler.get_scheduler()

    # FIXMATCH PAPER
    # Exponential moving average (EMA). We utilize EMA technique with decay 0.999.
    # The idea behind EMA is to maintain a separate set of "shadow" weights, 
    # which are updated as a weighted average of the current weights and their previous values. 
    # This can help to mitigate the effect of noisy updates and can make the training process more stable.
    ema_model = ModelEMA(model, device)

    # Print some useful information before starting training
    print(" =======> LET's GO <=======")
    print(f"  Num Epochs = {NUM_EPOCHS}")
    print(f"  Eval steps = {EVAL_STEPS}")
    print(f"  Batch size = {BATCH_SIZE}")

    # Zero out model gradients and start training
    model.zero_grad()
    train(labeled_data_loader, unlabeled_data_loader, test_data_loader, model, optimizer,
            ema_model, scheduler, tensor_board)


def train(labeled_data_loader, unlabeled_data_loader, test_data_loader, model,
          optimizer, ema_model, scheduler, tensor_board):
    """
    Description:
    This function trains the model using labeled and unlabeled data. It uses FixMatch algorithm for SSL.

    Inputs:
    - labeled_data_loader: Dataloader object containing labeled data.
    - unlabeled_data_loader: Dataloader object containing unlabeled data.
    - test_data_loader: Dataloader object containing test data.
    - model: PyTorch model object to train.
    - optimizer: PyTorch optimizer object to use for training.
    - ema_model: Exponential Moving Average Model object.
    - scheduler: Learning rate scheduler.
    - tensor_board: PyTorch SummaryWriter object to store logs for TensorBoard.

    Outputs: None
    """
    global best_accuracy
    test_accs = []

    # Initialize the iterators for labeled and unlabeled data loaders
    labeled_iter = iter(labeled_data_loader)
    unlabeled_iter = iter(unlabeled_data_loader)

    model.train()

    for epoch in range(NUM_EPOCHS):
        # Define the loss meters to keep track of the losses
        losses = AverageMeter()
        losses_labeled = AverageMeter()
        losses_unlabeled = AverageMeter()
        learning_rate = AverageMeter()

        # Initialize the progress bar for the current epoch
        p_bar = tqdm(range(EVAL_STEPS), desc=f'Train Epoch: {epoch+1}/{NUM_EPOCHS}')

        for batch_idx in range(EVAL_STEPS):
            # Get the labeled data and move it to the device
            try:
                inputs_labeled, targets_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_data_loader)
                inputs_labeled, targets_labeled = next(labeled_iter)

            # Get the unlabeled data and move it to the device
            try:
                (inputs_unlabeled_weak, inputs_unlabeled_strong), _ = next(unlabeled_iter)
            except :
                unlabeled_iter = iter(unlabeled_data_loader)
                (inputs_unlabeled_weak, inputs_unlabeled_strong), _ = next(unlabeled_iter)

            # This block of code prepares the input data for the model by concatenating the labeled, 
            # weakly-augmented and strongly-augmented unlabeled data samples.
            #  The data is then reshaped into a format that allows for the model 
            # to process both the labeled and unlabeled samples simultaneously.
            batch_size = inputs_labeled.shape[0]
            inputs = torch.cat((inputs_labeled, inputs_unlabeled_weak, inputs_unlabeled_strong))
            # Reshapes the concatenated tensor into a 3-dimensional tensor, where the first dimension represents the number of sections, 
            # the second dimension represents the number of samples in each section (i.e. labeled + 2 * MU), 
            # and the remaining dimensions represent the shape of each sample (e.g. 3 x 32 x 32 for CIFAR-10).
            inputs = inputs.view(-1, 2*MU+1, *inputs.shape[1:])
            inputs = inputs.permute(1, 0, *range(2, inputs.dim()))
            inputs = inputs.reshape(-1, *inputs.shape[2:]).to(device)

            # Move the labeled targets to the device
            targets_labeled = targets_labeled.to(device)

            # Forward pass through the model
            out_1 = model(inputs)

            # reshapes the output tensor out_1 obtained from the forward pass through the model. 
            # It first calculates the number of sections in the tensor by dividing the length of out_1 by (2*MU+1), where MU is a hyperparameter.
            # Then, it reshapes the tensor into 2*MU+1 sections along the first dimension, each of which has num_sections elements
            num_sections = len(out_1) // (2*MU+1)
            out_1 = out_1.view(2*MU+1, num_sections, *out_1.shape[1:])
            out_1 = out_1.permute(1, 0, *range(2, out_1.dim()))
            out_1 = out_1.reshape(-1, *out_1.shape[2:])

            # extracts the logits for the labeled data from the output of the model.
            logits_labeled = out_1[:batch_size]
            # extracts the logits for the unlabeled data from the output of the model.
            logits_unlabeled_weak, logits_unlabeled_strong = out_1[batch_size:].chunk(2)
            del out_1

            L_labeled = F.cross_entropy(logits_labeled, targets_labeled, reduction='mean')

            # This line calculates the soft labels for the weakly augmented images 
            # using the softmax function and detaching the output from the graph to avoid backpropagating through it.
            pseudo_label = torch.softmax(logits_unlabeled_weak.detach()/TEMPERATURE, dim=-1)
            # extracts the predicted labels for the weakly augmented image
            max_probs, targets_unlabeled = torch.max(pseudo_label, dim=-1)
            # This line creates a mask indicating which weakly augmented images 
            # have high enough confidence to be considered for training. It will grow over the training
            # THRESHOLD is a hyperparameter value is taken from the paper > 0.95 is good but <0.95 will be bad
            mask = max_probs.ge(THRESHOLD).float()

            # calculates the cross-entropy loss for the strongly augmented images, 
            # but only for those images that have high enough confidence based on the mask.
            L_unlabeled = (F.cross_entropy(logits_unlabeled_strong, targets_unlabeled,
                              reduction='none') * mask).mean()

            # calculates the overall loss for the batch using the labeled and unlabeled losses.
            # LAMBDA_U is a hyerparameter denoting the relative weight of the unlabeled loss
            loss = L_labeled + LAMBDA_U * L_unlabeled


            # Update Averagemeter() object
            losses.update(loss.item())
            losses_labeled.update(L_labeled.item())
            losses_unlabeled.update(L_unlabeled.item())
            learning_rate.update(scheduler.get_last_lr()[0])

            # Classic update for DNN
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema_model.update(model)
            model.zero_grad()

            if batch_idx % 1 == 0:
                p_bar.set_postfix({'LR': scheduler.get_last_lr()[0], 'Loss': losses.avg,
                                   'Loss_labeled': losses_labeled.avg,
                                   'Loss_unlabeled': losses_unlabeled.avg})
                p_bar.update()

        p_bar.close()

        # Evaluate model on test set and log results
        test_model = ema_model.ema
        test_loss, test_acc = test(test_data_loader, test_model)

        # Put the value in our tensorboard
        tensor_board.add_scalar('train/1.train_loss', losses.avg, epoch)
        tensor_board.add_scalar('train/2.train_loss_labeled', losses_labeled.avg, epoch)
        tensor_board.add_scalar('train/3.train_loss_unlabeled', losses_unlabeled.avg, epoch)
        tensor_board.add_scalar('train/4.learning_rate', learning_rate.avg, epoch)
        tensor_board.add_scalar('test/1.test_acc', test_acc, epoch)
        tensor_board.add_scalar('test/2.test_loss', test_loss, epoch)

        # Keep track of our best accuracy ie our best model
        is_best = test_acc > best_accuracy
        best_accuracy = max(test_acc, best_accuracy)

        # Save model checkpoint if it's the best so far
        model_to_save = model.module if hasattr(model, "module") else model
        ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
        # If training stop unexpectedly save scheduler optimizer state to restart
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_accuracy,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, RESULT_PATH)

        # Keep track of test accuracies for early stopping
        test_accs.append(test_acc)

    # Close TensorBoard writer
    tensor_board.close()

def test(test_data_loader, model):
    """
    Description: Performs a test pass through a PyTorch model and computes the average loss and accuracy over the test data
    Inputs:
        - test_data_loader: PyTorch DataLoader object containing the test data
        - model: PyTorch model object to test
    Outputs:
        - losses.avg: scalar value representing the average loss over the test data
        - accuracy_labeled.avg: scalar value representing the average accuracy over the test data for labeled examples
    """
    losses = AverageMeter()  # Create an AverageMeter object to track the loss
    accuracy_labeled = AverageMeter()  # Create an AverageMeter object to track the accuracy of labeled examples

    # Wrap the data loader with a progress bar
    test_data_loader = tqdm(test_data_loader, desc='Test')

    with torch.no_grad():
        # Iterate over the test data
        for batch_idx, (inputs_labeled, targets_labeled) in enumerate(test_data_loader):
            model.eval()  # Set the model to evaluation mode

            # Move the inputs and targets to the device
            inputs_labeled = inputs_labeled.to(device)
            targets_labeled = targets_labeled.to(device)

            # Forward pass through the model
            outputs_labeled = model(inputs_labeled)

            # Compute the loss and accuracy
            loss_labeled = F.cross_entropy(outputs_labeled, targets_labeled)
            acc_labeled = accuracy(outputs_labeled, targets_labeled)

            # Update the loss and accuracy meters
            losses.update(loss_labeled.item(), inputs_labeled.shape[0])
            accuracy_labeled.update(acc_labeled, inputs_labeled.shape[0])

            # Update the progress bar every 50 batches
            if batch_idx % 50 == 0:
                test_data_loader.set_postfix({'Loss': losses.avg, 'Acc_labeled': accuracy_labeled.avg})

        # Close the progress bar
        test_data_loader.close()

    return losses.avg, accuracy_labeled.avg



if __name__ == '__main__':
    main()
