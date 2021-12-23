# Implémentez un modèle de scoring:

## Présentation:

La mission principale de ce projet est de prédire le risque de faillite d'un client pour une société de crédit. Pour cela, nous devons configurer un modèle de classification binaire et d'en analyser les différentes métriques.

Ce projet consiste à créer une API web avec un Dashboard interactif. Celui-ci devra a minima contenir les fonctionnalités suivantes :

Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

## Construction
Dans ce dépôt, vous trouverez :

Un dossier notebook de preprocessing pour l'étude des données, l'entraînement et la configuration du modèle de classification.

Un fichier avec la configuration locale de l'API  api.py qui est le fichier Flask contenant la partie backend.

un dossier venv_7 qui repr♪0sente l'environnement virtuel créé pour ce projet.