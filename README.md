# Adversarial Training 

## PGD train

The codes can run with the following command:
Vanilla adversarial training. 
```
python pgdtrain.py --pgd
```
Low frequency training:
```
python pgdtrain.py --pgd --lowpass --fre [number of frequencies used]
```


### Reference
Towards Deep Learning Models Resistant to Adversarial Attacks
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu
https://arxiv.org/abs/1706.06083.

