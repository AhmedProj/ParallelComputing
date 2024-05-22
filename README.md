# ParallelComputing

## Etapes pour utiliser le package

1. Pour utiliser notre package il faut faire un clone du dépot au début :   
```git clone https://github.com/AhmedProj/ParallelComputing.git```

2. Se placer dans la racine du projet et cloner le depot de pybind11:
   
```cd ParallelComputing```

```git clone https://github.com/pybind/pybind11.git.```

4. Executer les commandes suivantes pour compiler le projet et pouvoir utiliser le module avec python:

```bash
mkdir build
cd build
cmake ..
make
