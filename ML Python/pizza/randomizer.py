# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

orders = np.random.randint(0, 25, size=(1000, 3))
orders = pd.DataFrame(data=orders, columns=['pizza1', 'pizza2', 'pizza3'])

mapping = {
1:"Mario Merola",
2:"Boticelli",
3:"Regina Margherita",
4:"Laura Pausini",
5:"Dario Argento",
6:"Alberto Sordi",
7:"De Sica",
8:"Marcello Mastroianni",
9:"Gigi D'Allesio",
10:"Massimo Troisi",
11:"Raffaello",
12:"Vanvitelli",
13:"Vittorio Gassman",
14:"Michilangelo",
15:"Vesuvio",
16:"Gianni Morandi",	
17:"Sophia Loren",
18:"Toto",
19:"Donatello",
20:"Pavarotti",
21:"Rafaella",
22:"Dante",
23:"Federico Fellini",
24:"Pino Daniele",
25:"Gioto",
0: ""}
orders_mapped = orders.copy()
for c in orders.columns:
    orders_mapped[str(c)] = orders[str(c)].map(mapping)
    
cost_mapping = {
1: 8,
2: 8.5,
3: 9.5,	
4: 10,
5: 10,	
6: 10,
7: 10,
8: 10,	
9: 10.5,
10: 10.5,	
11: 10.5,
12: 11.5,	
13: 11.5,	
14: 11.5,
15: 11.5,
16: 12,	
17: 12,
18: 12,
19: 12,	
20: 12.5,
21: 12.5,
22: 12.5,
23: 12.5,	
24: 13,
25: 13,	
0: 0}

orders_cost = orders.copy()
for c in orders.columns:
    orders_cost[str(c)] = orders[str(c)].map(cost_mapping)
orders_cost['total'] = orders_cost['pizza1'] + orders_cost['pizza2'] + orders_cost['pizza3']

orders_concat = pd.Series(np.zeros((1000)))
orders_list = []
for i in range(len(orders_mapped)):
    temp=[]
    for c in orders_mapped.columns:
        if orders_mapped.loc[i, str(c)] == '':
            pass
        else:
            temp.append(orders_mapped.loc[i, str(c)])
    orders_list.append(temp)
    del(temp)
orders_concat = pd.Series(orders_list)
for i in range(len(orders_concat)):
    words = ','.join(str(w) for w in orders_concat[i])
    orders_concat[i] = words
final = pd.DataFrame(data=np.zeros(shape=(1000, 1)), columns=['orders'])
final['orders'] = orders_concat.values
final['total_amount'] = orders_cost['total']
final.to_csv('pizza_dataset.csv', index=False, sep=';')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    