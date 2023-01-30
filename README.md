# textures-cellular-automata

Implémentation de [2]

Projet MVA S1 pour J. Delon, Y. Gousseau

[Descriptif du projet](https://perso.telecom-paristech.fr/gousseau/MVA/Projets2022/TextureAC/)

## Détails d'implémentation (et différences avec [2])

**Extraction des texture features avec VGG-16 :**

Dans le principe, on suit la méthode avec ~10k paramètres de [1, §5], c'est-à-dire utiliser les 5 couches conv1_1 et pool1-4, et faire une PCA sur les channels de chaque couche pour n'en garder que 64 par couche. On calcule ensuite les matrices de Gram (corrélations entre channels) pour avoir des features invariantes.

[2] n'expliquent pas ces détails, et (d'après leur code) procèdent différemment :
- les couches choisies sont un peu différentes (2 couches plus loin, sauf la première qui est une couche plus loin)
- les projections pour réduire le nombre de channels par couche ne sont *pas* des PCA, elles sont aléatoires !
- l'étape de calcul de matrices de Gram est supprimée (ce ne sont donc plus des "space invariant features" ?!)

Rmq: la PCA coûte trop cher, on fait aussi des projections aléatoires

## À faire

- training loop
- tester !

## Références

[1] Gatys, Ecker, Bethge, Texture Synthesis Using Convolutional Neural Networks

[2] Mordvintsev, Niklasson, Randazzo, Texture Generation with Neural Cellular Automata
