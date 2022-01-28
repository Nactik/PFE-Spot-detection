# PFE SPOT DETECTION

## Requierements:

- Ubuntu 18.04 LTS
- Python 3.6.9
- SDK Spot by Boston Dynamics : [Installation](https://dev.bostondynamics.com/docs/python/quickstart)
- YoloV5 repository : [GitHub](https://github.com/ultralytics/yolov5)

## Installation

### Clonage des répo

Cloner le repo actuel et le repo de yoloV5 dans un même dossier selon l'architecture suivante :

```bash
Current folder
├── PFE-Spot-detection
│   ├── ...
│   ├── ...
├── yolov5
│   ├── ...
│   ├── ...
├── spot-sdk-master
│   ├── ...
│   ├── ...
```

### Installation des dépendances

Depuis le dossier racine, naviguer vers le dossier `PFE-Spot detection` et installer les dépendances :

```bash
cd PFE-Spot-detecion
python3 -m pip install -r requierements.txt
```

### Configuration de l'environnement

Afin de pouvoir se connecter au robot spot, il est nécessaire créer un fichier  nommé `.env` à la racine de ce répo et de compléter les informations de connextion selon le format suivant :

```bash
ROBOT_USERNAME=user
ROBOT_PASSWORD=password
ROBOT_IP=XXX.XXX.XXX.XXX
```

## Utilisation

Afin de pouvoir utilisé le programme, il faut d'abord être connecté au point d'accès wifi du robot spot (si utilisation sur pc distant)

### Lancement du progamme

Ouvrir 2 terminaux differents : 

#### Premier terminal :

Le premier terminal va servir à prendre le controle de l'estop:

- Naviguer vers le dossier `spot-sdk-master/python/examples/estop`
```bash
cd spot-sdk-master/python/examples/estop
```

- [OPTIONNEL] Si ce n'est pas déjà fait, installer les requierements :

```bash
python3 -m pip install -r requierements.txt
```

- Lancer le programme esttop-nogui en remplacement les valeurs de user, password et IP

```bash
python3 estop-nogui.py --username <user> --password <password> <IP> 
```

#### Second terminal

Ce terminal va permettre de lancer le programme principale : 

- Naviguer vers la racine du dossier `PFE-Spot-infive`

```bash
cd PFE-Spot-infive
```
- Executer le programme avec la commande suivante :

```bash
python3 command-line.py
```

- Pour quitter le programme, appuyer sur CTRL-C




