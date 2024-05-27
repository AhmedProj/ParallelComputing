# ParallelComputing

## Prerequisites
1. Install Cmake On Linux :

```sudo apt update; sudo apt install cmake```
\```python
def hello_world():
    print("Hello, world!")
\```

2. GPU device since first computations en Llama needs it. 

## Steps tu use our package in Linux system

1. To use our package you have first to clone the repository :
   
```git clone https://github.com/AhmedProj/ParallelComputing.git```

2. Then change directory and install requirements for python :  
   
```cd ParallelComputing```  

```pip install -r requirements.txt```  

3. Compile the project and make the python package accessible from any directory:

```python
   python setup.py install
