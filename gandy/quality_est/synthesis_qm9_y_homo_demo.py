{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "import deepchem as dc\n",
    "import numpy as np\n",
    "import random\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt\n",
    "import tensorflow as tf\n",
    "from pandas import Series, DataFrame\n",
    "\n",
    "from deepchem.molnet import load_qm9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "qm9_tasks, datasets, transformers = load_qm9()\n",
    "train_dataset, valid_dataset, test_dataset = datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(<DiskDataset X.shape: (105984, 29, 29), y.shape: (105984, 12), w.shape: (105984, 12), task_names: ['mu' 'alpha' 'homo' ... 'u298' 'h298' 'g298']>,\n",
       " <DiskDataset X.shape: (13248, 29, 29), y.shape: (13248, 12), w.shape: (13248, 12), task_names: ['mu' 'alpha' 'homo' ... 'u298' 'h298' 'g298']>,\n",
       " <DiskDataset X.shape: (13248, 29, 29), y.shape: (13248, 12), w.shape: (13248, 12), task_names: ['mu' 'alpha' 'homo' ... 'u298' 'h298' 'g298']>)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XR = test_dataset.X.reshape((-1, 29*29))\n",
    "type(test_dataset.X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        15.840093\n",
       "1        17.937048\n",
       "2        12.523278\n",
       "3        20.987048\n",
       "4        12.772426\n",
       "           ...    \n",
       "13243    12.376352\n",
       "13244    15.352393\n",
       "13245    12.121365\n",
       "13246    21.162649\n",
       "13247    16.220116\n",
       "Name: 1, Length: 13248, dtype: float64"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = DataFrame(XR, index = [list(range(0,13248))])\n",
    "df[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['mu',\n",
       " 'alpha',\n",
       " 'homo',\n",
       " 'lumo',\n",
       " 'gap',\n",
       " 'r2',\n",
       " 'zpve',\n",
       " 'cv',\n",
       " 'u0',\n",
       " 'u298',\n",
       " 'h298',\n",
       " 'g298']"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "qm9_tasks #show all the dimensions of y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "#extrct the 'homo'values\n",
    "Y = test_dataset.y\n",
    "YT = Y.T\n",
    "a = np.zeros ([12,13248])\n",
    "for i in range(12):\n",
    "    a[i] = YT[i]\n",
    "y_homo = a[2]\n",
    "y = y_homo.tolist()\n",
    "l = y_homo.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "x1 = np.random.uniform(0, l, 1).astype(np.int) #set the number of noise added\n",
    "x2 = np.random.uniform(0, l, x1) #x1 numbers of random value\n",
    "a = len(x2)\n",
    "f = np.zeros(l) #defined to save noise values \n",
    "x = np.zeros(l) #defined to save x2 values\n",
    "\n",
    "#add noise to x1 numbers of y\n",
    "for i in range(a):\n",
    "    mu = 0\n",
    "    sigma = (x1+x2)/2\n",
    "    noise = np.random.normal(mu, np.abs(sigma), x1)\n",
    "    g = noise.tolist()\n",
    "    y[i] += g[i]\n",
    "    f[i] += g[i]\n",
    "    x[i] = x2[i] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid character in identifier (<ipython-input-73-24cdcf00ae8a>, line 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-73-24cdcf00ae8a>\"\u001b[0;36m, line \u001b[0;32m2\u001b[0m\n\u001b[0;31m    dataframe = pd.DataFrame({'x1':df[0],'x2':df[1],'x3':df[2],'x4':df[3],'x5':df[4],'x6':df[5],'x7':df[6],'x8':df[7],'x9':df[8],'x10':df[9],'x11':df[10],'x12':df[11],'x13':df[12],'x14':df[13],'x15':df[14],'x16':df[15],'x17':df[16],'x18':df[17],'x19':df[18],'x20':df[19],'x21':df[20],'x22':df[21],'x23':df[22],'x24':df[23],'x25':df[24],'x26':df[25],'x27':df[26]，'x28':df[27],'x29':df[28],'noise':f, 'y_homo_initial':y_homo,'y_homo_synthetic':y})\u001b[0m\n\u001b[0m                                                                                                                                                                                                                                                                                                                                                                         ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid character in identifier\n"
     ]
    }
   ],
   "source": [
    "#save to_csv:\n",
    "dataframe = pd.DataFrame({'x1':df[0],'x2':df[1],'x3':df[2],'x4':df[3],'x5':df[4],'x6':df[5],'x7':df[6],'x8':df[7],'x9':df[8],'x10':df[9],'x11':df[10],'x12':df[11],'x13':df[12],'x14':df[13],'x15':df[14],'x16':df[15],'x17':df[16],'x18':df[17],'x19':df[18],'x20':df[19],'x21':df[20],'x22':df[21],'x23':df[22],'x24':df[23],'x25':df[24],'x26':df[25],'x27':df[26]，'x28':df[27],'x29':df[28],'noise':f, 'y_homo_initial':y_homo,'y_homo_synthetic':y})\n",
    "\n",
    "#dataframe.to_csv(\"y_homo_data.csv\", index=False, sep = ',')\n",
    "#mi.to_flat_index().union(other)\n",
    "#DF = pd.concat([df,dataframe],axis=1)\n",
    "#gen_data = pd.read_csv('y_homo_data.csv')\n",
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
