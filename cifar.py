import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random
import math

# Inspired by
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def obtain_cifar10_dataset(data_root):
    """
    Description: Cette fonction retourne trois objets CIFAR10SemiSupervised : un objet d'entraînement étiqueté, un objet d'entraînement non étiqueté 
    et un objet de test. Les données CIFAR10 sont téléchargées et stockées localement si nécessaire.
    Entrée: data_root (chaîne de caractères) - Le chemin du répertoire où les données CIFAR10 sont stockées.
    Sortie: Un tuple de trois objets CIFAR10SemiSupervised - l'objet d'entraînement étiqueté, l'objet d'entraînement non étiqueté et l'objet de test.
    """
    # Récupération de nos données
    base_data = datasets.CIFAR10(data_root, train=True, download=True)
    # les labels de nos images labelisés, 50 000 labels non labelisés, je ne soustrait pas les labelisés
    labeled_indices, unlabeled_indices = split_indices(base_data.targets)
    print("label", labeled_indices.shape, unlabeled_indices.shape)

    # transformation + normalisation
    labeled_transformations = build_labeled_transforms()
    validation_transformations = build_validation_transforms()

    # on crée des objets CIFAR 10 pour nos 3 données labelisés, non labelisés et le test
    # pour chacun on envoie la transformation a effectué sur nos images
    labeled_train_data = CIFAR10SemiSupervised(
        data_root, labeled_indices, train=True, transform=labeled_transformations)

    unlabeled_train_data = CIFAR10SemiSupervised(
        data_root, unlabeled_indices, train=True,
        transform=lambda x: transform_fix_match_ssl(x, mean=cifar10_mean, std=cifar10_std)) # il y a deux transformations a effectué weak et strong

    test_data = datasets.CIFAR10(
        data_root, train=False, transform=validation_transformations, download=False)
    print("Number of labeled examples: ", len(labeled_train_data))
    print("Number of unlabeled examples: ", len(unlabeled_train_data))
    print("Number of test examples: ", len(test_data))
    return labeled_train_data, unlabeled_train_data, test_data

def build_labeled_transforms():
    """
    Description: Cette fonction retourne une liste de transformations d'images appliquées sur les images étiquetées pour l'entraînement.
    Entrée: Aucune
    Sortie: Une liste de transformations d'images (transforms.Compose)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

def build_validation_transforms():
    """
    Description: Cette fonction retourne une liste de transformations d'images appliquées sur les images pour la validation.
    Entrée: Aucune
    Sortie: Une liste de transformations d'images (transforms.Compose)
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])


# Define a function that takes a list of labels and returns two numpy arrays - one for labeled data indices and one for unlabeled data indices.
def split_indices(label_list):
    """
    Description: This function takes a list of labels and returns two numpy arrays - one for labeled data indices and one for unlabeled data indices.
    Input: label_list (list) - The list of labels for all CIFAR10 data.
    Output: A tuple of two numpy arrays - the array of indices for labeled data, and the array of indices for unlabeled data.
    """
    # Define variables that we will use
    n_classes = 10
    per_class_labels = 250 // n_classes
    label_list = np.array(label_list)

    # Select a fixed number of labels per class for labeled data, in our case 25 by labels
    labeled_indices = np.array([
        idx for i in range(n_classes) for idx in np.random.choice(
            np.where(label_list == i)[0], per_class_labels, replace=False)
    ])
    
    # Create an array of indices for all data (labeled and unlabeled)
    unlabeled_indices = np.array(range(len(label_list)))
    
    # Shuffle the labeled data indices for better randomness
    np.random.shuffle(labeled_indices)
    
    # Print the shape of the labeled indices for debugging purposes
    print(labeled_indices.shape)
    
    # Return the labeled and unlabeled indices as a tuple
    return labeled_indices, unlabeled_indices


def transform_fix_match_ssl(x, mean, std):
    """
    Description: Cette fonction prend une image x ainsi que la moyenne et l'écart-type des données CIFAR10 
    et retourne une paire d'images 
    - une version transformée faible et une version transformée forte de l'image d'entrée.
    Entrée:
    x (PIL Image) - L'image d'entrée.
    mean (tuple) - Le tuple de la moyenne des données CIFAR10.
    std (tuple) - Le tuple de l'écart-type des données CIFAR10.
    Sortie: Un tuple d'images PIL - une image faible et une image forte.
    """
    weak_transforms = build_labeled_transforms()
    strong_transforms = create_strong_transforms(mean, std)
    return weak_transforms(x), strong_transforms(x)

def create_strong_transforms(mean, std):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=int(32*0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10), # Strong augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])


class CIFAR10SemiSupervised(datasets.CIFAR10):
    """
    Description: Cette classe hérite de la classe datasets.CIFAR10 et est utilisée pour définir les ensembles d'entraînement étiquetés et non étiquetés.
    Attributs:
        root (chaîne de caractères) - Le chemin du répertoire où les données CIFAR10 sont stockées.
        indexs (tableau numpy) - Le tableau d'indices pour les données étiquetées et non étiquetées.
        train (bool) - Si l'objet de données est utilisé pour l'entraînement ou non.
        transform (list) - La liste des transformations à appliquer aux données.
        download (bool) - Si les données doivent être téléchargées ou non.
        data (tableau numpy) - Les données CIFAR10.
        targets (tableau numpy) - Les étiquettes pour les données CIFAR10.
    Méthodes:
        __getitem__(self, index) - Cette méthode retourne une paire d'image et d'étiquette pour un index donné. 
            L'image est transformée à l'aide des transformations spécifiées lors de la création de l'objet.
        __len__(self) - Cette méthode retourne le nombre d'exemples dans l'ensemble de données.

    """
    def __init__(self, root, indexs, train=True, transform=None, download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.transform(Image.fromarray(img))

        return img, target



class RandAugmentations:
    def __init__(self):
        pass
    
    # Helper function to scale a float parameter to the appropriate range
    def _float_parameter(self, v, max_v):
        return float(v) * max_v / 10
    
    # Helper function to scale an int parameter to the appropriate range
    def _int_parameter(self, v, max_v):
        return int(v * max_v / 10)
    
    # Apply the AutoContrast transformation to an image
    def AutoContrast(self, img, **kwarg):
        return PIL.ImageOps.autocontrast(img)
    
    # Adjust the brightness of an image
    def Brightness(self, img, v, max_v):
        v = self._float_parameter(v, max_v) + 0.05
        return PIL.ImageEnhance.Brightness(img).enhance(v)
    
    # Adjust the color of an image
    def Color(self, img, v, max_v):
        v = self._float_parameter(v, max_v) + 0.05
        return PIL.ImageEnhance.Color(img).enhance(v)
    
    # Adjust the contrast of an image
    def Contrast(self, img, v, max_v):
        v = self._float_parameter(v, max_v) + 0.05
        return PIL.ImageEnhance.Contrast(img).enhance(v)
    
    # Cut out a rectangular portion of an image
    def CutoutAbs(self, img, v, **kwarg):
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        xy = (x0, y0, x1, y1)
        # gray
        color = (127, 127, 127)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img  
    
    # Apply the Equalize transformation to an image
    def Equalize(self,img, **kwarg):
        return PIL.ImageOps.equalize(img)
    
    # Return the original image unchanged
    def Identity(self,img, **kwarg):
        return img
    
    # Posterize an image to reduce the number of color levels
    def Posterize(self,img, v, max_v):
        v = self._int_parameter(v, max_v) + 4
        return PIL.ImageOps.posterize(img, v)
    
    # Rotate an image by a random angle
    def Rotate(self,img, v, max_v, bias=0):
        v = self._int_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        return img.rotate(v)
    
    # Increase the sharpness of an image
    def Sharpness(self,img, v, max_v):
        v = self._float_parameter(v, max_v) + 0.05
        return PIL.ImageEnhance.Sharpness(img).enhance(v)


    # Define the ShearX function which applies horizontal shear to an image
    def ShearX(self,img, v, max_v, bias=0):
        # Choose a random float value 'v' between 0 and max_v and add bias to it
        v = self._float_parameter(v, max_v) + bias
        # Flip the sign of 'v' randomly with a probability of 0.5
        if random.random() < 0.5:
            v = -v
        # Apply the affine transformation to the image with a 1x2 matrix that shears it horizontally
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

    # Define the ShearY function which applies vertical shear to an image
    def ShearY(self,img, v, max_v, bias=0):
        # Choose a random float value 'v' between 0 and max_v and add bias to it
        v = self._float_parameter(v, max_v) + bias
        # Flip the sign of 'v' randomly with a probability of 0.5
        if random.random() < 0.5:
            v = -v
        # Apply the affine transformation to the image with a 1x2 matrix that shears it vertically
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

    # Define the Solarize function which inverts the pixel values of an image above a certain threshold
    def Solarize(self,img, v, max_v, bias=0):
        # Choose a random integer value 'v' between 0 and max_v and add bias to it
        v = self._int_parameter(v, max_v) + bias
        # Invert the pixel values of the image above the threshold of (256 - v)
        return PIL.ImageOps.solarize(img, 256 - v)

    # Define the TranslateX function which translates an image horizontally
    def TranslateX(self,img, v, max_v):
        # Choose a random float value 'v' between 0 and max_v
        v = self._float_parameter(v, max_v) 
        # Flip the sign of 'v' randomly with a probability of 0.5
        if random.random() < 0.5:
            v = -v
        # Scale 'v' by the width of the image and convert it to an integer
        v = int(v * img.size[0])
        # Apply the affine transformation to the image with a 1x2 matrix that translates it horizontally
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    # Define the TranslateY function which translates an image vertically
    def TranslateY(self,img, v, max_v):
        # Choose a random float value 'v' between 0 and max_v
        v = self._float_parameter(v, max_v) 
        # Flip the sign of 'v' randomly with a probability of 0.5
        if random.random() < 0.5:
            v = -v
        # Scale 'v' by the height of the image and convert it to an integer
        v = int(v * img.size[1])
        # Apply the affine transformation to the image with a 1x2 matrix that translates it vertically
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)) 

    # Define the fixmatch_augment_pool function which returns a list of image augmentation functions with their parameters
    def fixmatch_augment_pool(self):
        # List of image augmentation functions and their parameters
        augs = [(self.AutoContrast, None),
                (self.Brightness, 0.9),
                (self.Color, 0.9),
                (self.Contrast, 0.9),            
                (self.Equalize, None),
                (self.Identity, None),
                (self.Posterize, 4),
                (self.Rotate, 30),
                (self.Sharpness, 0.9),
                (self.ShearX, 0.3),
                (self.ShearY, 0.3),
                (self.Solarize, 256),
                (self.TranslateX, 0.3),
                (self.TranslateY, 0.3)]
        # Return the list of image augmentation functions and their parameters
        return augs

    
# Define the RandAugmentMC class which applies a random set of image augmentations to an image
class RandAugmentMC:
    def __init__(self, n, m):
        # Initialize the number of transformations to apply (n) and the maximum magnitude of each transformation (m)
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        # Create an instance of the RandAugmentations class which contains the image augmentation functions
        self.rand_augmentations = RandAugmentations()
        # Generate a pool of possible image transformations using the fixmatch_augment_pool method from the RandAugmentations class
        self.transform_pool = self.rand_augmentations.fixmatch_augment_pool()

    def apply_transform(self, image, operation, max_val):
        # Apply a random image transformation to the input image
        # Choose a random magnitude value between 1 and m
        value = np.random.randint(1, self.m)
        # Apply the chosen operation to the image with the randomly chosen magnitude and the maximum magnitude value
        if random.random() < 0.5:
            image = operation(image, v=value, max_v=max_val)
        # Return the transformed image
        return image

    def process_image(self, img):
        # Apply a random set of n image transformations to the input image
        # Choose n image transformations randomly from the transform_pool
        chosen_transforms = random.choices(self.transform_pool, k=self.n) 
        for transform, max_value in chosen_transforms:
            # Apply the chosen transformation to the input image with a random magnitude between 1 and m
            img = self.apply_transform(img, transform, max_value)
        # Apply CutoutAbs operation to the image to remove part of it
        img = self.rand_augmentations.CutoutAbs(img, int(32 * 0.5)) 
        # Return the final transformed image
        return img

    def __call__(self, img):
        # This method is called when the instance of this class is called with an input image as an argument
        # It simply calls the process_image method to apply a random set of image augmentations to the input image and returns the transformed image
        return self.process_image(img)