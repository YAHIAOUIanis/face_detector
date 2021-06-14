# FACE DETECTOR

### Quelques commandes basiques de Git :

Une fois qu'un dépôt est cloné, vous travaillerez à l'intérieur de la branche par défaut (la valeur par défaut est `main`)
```
git clone https://github.com/YAHIAOUIanis/face_detector.git && cd face_detector
```
Commande de base, obtenir les modifications de la branche actuelle vers le référentiel distant
```
git pull
```
Pour ajouter un fichier afin de le commiter
```
git add <file>
```
Ou pour ajouter tous les fichiers
```
git add -A 
```
Commitez un instantané de tous les changements apportés au répertoire de travail (ceux qui ont été ajoutés grâce à git add à un moment donné dans l'historique).
```
git commit -m "ton message"
```
Pousser les changements dans la branche `main` vers github maître d'origine git push
```
git push origin main
```