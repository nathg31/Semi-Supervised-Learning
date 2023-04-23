import torch
from copy import deepcopy
import shutil 
import os

class ModelEMA(object):
    """
    Description: Cette classe est utilisée pour maintenir une moyenne exponentielle mobile (EMA) 
    des poids du modèle en vue d'une meilleure généralisation et d'une réduction de la variance des poids lors de l'entraînement.
    Cette classe stocke le modèle original ainsi que le modèle EMA.
    Attributs:
        ema (torch.nn.Module) - Le modèle EMA.
        decay (float) - Le coefficient de décroissance de la moyenne mobile exponentielle.
        ema_has_module (bool) - Vrai si self.ema contient un attribut module (si le modèle a été entraîné en utilisant le multiprocessing).
        param_keys (list) - Une liste des noms des paramètres de self.ema.
        buffer_keys (list) - Une liste des noms des tampons de self.ema.
    Méthodes:
        __init__(self, model, device, decay=0.999) - Initialise un objet ModelEMA avec un modèle model, un périphérique device et un coefficient de décroissance decay.
        update(self, model) - Met à jour le modèle EMA à partir du modèle model.
    """
    def __init__(self, model, device, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        Description: Cette fonction prend en entrée une sortie de modèle et les étiquettes cibles correspondantes 
        et calcule la précision du modèle.
        Entrée:
            output (torch.Tensor) - La sortie du modèle.
            target (torch.Tensor) - Les étiquettes cibles correspondantes.
        Sortie: La précision du modèle (float).
        """
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262

    Description: Cette classe est utilisée pour stocker et calculer la moyenne d'une valeur 
        (comme la perte ou la précision) sur plusieurs itérations.
    Attributs:
        val (float) - La valeur de la dernière itération.
        avg (float) - La moyenne de toutes les itérations précédentes.
        sum (float) - La somme de toutes les valeurs des itérations précédentes.
        count (int) - Le nombre total d'itérations précédentes.
    Méthodes:
        __init__(self) - Initialise un objet AverageMeter.
        reset(self) - Réinitialise l'objet AverageMeter.
        update(self, val, n=1) - Met à jour l'objet AverageMeter avec la nouvelle valeur val.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # pour chaque nouvelle valeur on update notre moyenne
        # on prends en compte le batch size
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    """
    Description: Saves the checkpoint of the current training state, including the model weights, optimizer state, and current epoch
    Inputs:
        - state: dictionary containing the current state of the training, including model weights, optimizer state, and current epoch
        - is_best: boolean value indicating whether the current state is the best performing so far
        - checkpoint: directory where the checkpoint should be saved
        - filename: name of the checkpoint file (default is 'checkpoint.pth.tar')
    Outputs: None
    """
    # Create the full filepath for the checkpoint file
    filepath = os.path.join(checkpoint, filename)
    # Save the state dictionary to the checkpoint file
    torch.save(state, filepath)
    # If this is the best performing state, save a copy of the checkpoint file with a different name
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def accuracy(output, target):
    """
    Description: Computes the accuracy of a model's predictions given the output logits and target labels
    Inputs:
        - output: tensor of shape (batch_size, num_classes) containing the logits for each example in the batch
        - target: tensor of shape (batch_size,) containing the target class for each example in the batch
    Outputs:
        - accuracy: scalar value representing the accuracy of the model's predictions on the batch, as a percentage
    """
    batch_size = target.size(0)  # Get the number of examples in the batch

    _, pred = torch.max(output, dim=1)  # Get the predicted class for each example in the batch
    correct = pred.eq(target).sum().item()  # Count the number of correct predictions

    accuracy = 100.0 * correct / batch_size  # Compute the accuracy as a percentage
    return accuracy
