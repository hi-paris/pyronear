import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from hyperopt import hp, fmin, tpe, Trials
from ultralytics import YOLO

# Définir l'espace de recherche des hyperparamètres
space = {
    'rotate': hp.uniform('rotate', -10, 10),
    'shift_x': hp.uniform('shift_x', -0.1, 0.1),
    'shift_y': hp.uniform('shift_y', -0.1, 0.1),
    'shear': hp.uniform('shear', -5, 5),
    'zoom': hp.uniform('zoom', 0.9, 1.1),
    'horizontal_flip': hp.choice('horizontal_flip', [0.0, 1.0]),
    'vertical_flip': hp.choice('vertical_flip', [0.0, 1.0]),
}

# Charger et préparer l'image
img = cv2.imread('image_test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir BGR à RGB
img_height, img_width, _ = img.shape

# Charger le modèle YOLOv10
weights_path = 'random_search/yolov10n.pt'
model = YOLO(weights_path)

# Fonction objectif pour l'optimisation
def objective(params):
    # Créer la transformation albumentation à partir des hyperparamètres
    transform = A.Compose([
        A.Rotate(limit=params['rotate'], p=1.0),
        A.ShiftScaleRotate(
            shift_limit=params['shift_x'], 
            scale_limit=(params['zoom'], params['zoom']), 
            rotate_limit=0, 
            p=1
        ),
        A.ShearX(shift_limit=params['shear'], p=1.0),
        A.HorizontalFlip(p=params['horizontal_flip']),
        A.VerticalFlip(p=params['vertical_flip']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], p=1.0)

       # Appliquer la transformation albumentation à l'image
    img_transformed = transform(image=img)['image']
    img_transformed = np.transpose(img_transformed, (2, 0, 1))
    img_transformed = torch.from_numpy(img_transformed).unsqueeze(0)  


    # Effectuer l'inférence avec le modèle YOLOv10
    with torch.no_grad():
        pred = model(img_transformed)
        pred = pred[0]  # supprimer la dimension batch

  # Calculer la perte du modèle sur l'image transformée
    target = torch.zeros((1, img_height // 32, img_width // 32, 85))  
    loss = model.loss(pred, target)
    return loss.item()

# Optimisation des hyperparamètres
num_iter = 50
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=num_iter, trials=trials)

print('Meilleurs hyperparamètres:', best)

# Appliquer les meilleures transformations et afficher les résultats
transform = A.Compose([
    A.Rotate(limit=best['rotate'], p=1.0),
    A.ShiftScaleRotate(
        shift_limit=best['shift_x'], 
        scale_limit=(best['zoom'], best['zoom']),  # Utiliser un tuple pour scale_limit
        rotate_limit=0, 
        p=1
    ),
    A.ShearX(shift_limit=best['shear'], p=1.0),
    A.HorizontalFlip(p=best['horizontal_flip']),
    A.VerticalFlip(p=best['vertical_flip']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], p=1.0)

img_transformed = transform(image=img)['image']
img_transformed = torch.unsqueeze(img_transformed, 0)  # Ajouter une dimension batch

with torch.no_grad():
    pred = model(img_transformed)[0]  # Obtenir les prédictions

# Préparer l'image pour l'affichage
img_display = np.transpose(img_transformed.squeeze().cpu().numpy(), (1, 2, 0)) 
img_display = np.clip(img_display * 255, 0, 255).astype(np.uint8) 

cv2.imshow('Detection results', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

