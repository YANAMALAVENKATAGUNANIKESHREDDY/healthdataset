{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "9abc257b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as npy\n",
    "import pandas as pds\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn import svm\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.preprocessing import LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9e5dc1a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "kd_dataset=pds.read_csv(\"D:\\Python Projects\\Guna\\Guna\\Datasets Zip File\\kidney.zip\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e1ed3a12",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>age</th>\n",
       "      <th>bp</th>\n",
       "      <th>sg</th>\n",
       "      <th>al</th>\n",
       "      <th>su</th>\n",
       "      <th>rbc</th>\n",
       "      <th>pc</th>\n",
       "      <th>pcc</th>\n",
       "      <th>ba</th>\n",
       "      <th>...</th>\n",
       "      <th>pcv</th>\n",
       "      <th>wc</th>\n",
       "      <th>rc</th>\n",
       "      <th>htn</th>\n",
       "      <th>dm</th>\n",
       "      <th>cad</th>\n",
       "      <th>appet</th>\n",
       "      <th>pe</th>\n",
       "      <th>ane</th>\n",
       "      <th>classification</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>48.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>1.020</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>normal</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>...</td>\n",
       "      <td>44</td>\n",
       "      <td>7800</td>\n",
       "      <td>5.2</td>\n",
       "      <td>yes</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>good</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>ckd</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>1.020</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>normal</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>...</td>\n",
       "      <td>38</td>\n",
       "      <td>6000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>good</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>ckd</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>62.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>1.010</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>normal</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>...</td>\n",
       "      <td>31</td>\n",
       "      <td>7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>no</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>poor</td>\n",
       "      <td>no</td>\n",
       "      <td>yes</td>\n",
       "      <td>ckd</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>48.0</td>\n",
       "      <td>70.0</td>\n",
       "      <td>1.005</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>abnormal</td>\n",
       "      <td>present</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>...</td>\n",
       "      <td>32</td>\n",
       "      <td>6700</td>\n",
       "      <td>3.9</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>poor</td>\n",
       "      <td>yes</td>\n",
       "      <td>yes</td>\n",
       "      <td>ckd</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>51.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>1.010</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>normal</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>notpresent</td>\n",
       "      <td>...</td>\n",
       "      <td>35</td>\n",
       "      <td>7300</td>\n",
       "      <td>4.6</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>good</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>ckd</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   id   age    bp     sg   al   su     rbc        pc         pcc          ba  \\\n",
       "0   0  48.0  80.0  1.020  1.0  0.0     NaN    normal  notpresent  notpresent   \n",
       "1   1   7.0  50.0  1.020  4.0  0.0     NaN    normal  notpresent  notpresent   \n",
       "2   2  62.0  80.0  1.010  2.0  3.0  normal    normal  notpresent  notpresent   \n",
       "3   3  48.0  70.0  1.005  4.0  0.0  normal  abnormal     present  notpresent   \n",
       "4   4  51.0  80.0  1.010  2.0  0.0  normal    normal  notpresent  notpresent   \n",
       "\n",
       "   ...  pcv    wc   rc  htn   dm  cad appet   pe  ane classification  \n",
       "0  ...   44  7800  5.2  yes  yes   no  good   no   no            ckd  \n",
       "1  ...   38  6000  NaN   no   no   no  good   no   no            ckd  \n",
       "2  ...   31  7500  NaN   no  yes   no  poor   no  yes            ckd  \n",
       "3  ...   32  6700  3.9  yes   no   no  poor  yes  yes            ckd  \n",
       "4  ...   35  7300  4.6   no   no   no  good   no   no            ckd  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kd_dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5dbf0ae3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(400, 26)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kd_dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "fe1f994a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>age</th>\n",
       "      <th>bp</th>\n",
       "      <th>sg</th>\n",
       "      <th>al</th>\n",
       "      <th>su</th>\n",
       "      <th>bgr</th>\n",
       "      <th>bu</th>\n",
       "      <th>sc</th>\n",
       "      <th>sod</th>\n",
       "      <th>pot</th>\n",
       "      <th>hemo</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>400.000000</td>\n",
       "      <td>391.000000</td>\n",
       "      <td>388.000000</td>\n",
       "      <td>353.000000</td>\n",
       "      <td>354.000000</td>\n",
       "      <td>351.000000</td>\n",
       "      <td>356.000000</td>\n",
       "      <td>381.000000</td>\n",
       "      <td>383.000000</td>\n",
       "      <td>313.000000</td>\n",
       "      <td>312.000000</td>\n",
       "      <td>348.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>199.500000</td>\n",
       "      <td>51.483376</td>\n",
       "      <td>76.469072</td>\n",
       "      <td>1.017408</td>\n",
       "      <td>1.016949</td>\n",
       "      <td>0.450142</td>\n",
       "      <td>148.036517</td>\n",
       "      <td>57.425722</td>\n",
       "      <td>3.072454</td>\n",
       "      <td>137.528754</td>\n",
       "      <td>4.627244</td>\n",
       "      <td>12.526437</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>115.614301</td>\n",
       "      <td>17.169714</td>\n",
       "      <td>13.683637</td>\n",
       "      <td>0.005717</td>\n",
       "      <td>1.352679</td>\n",
       "      <td>1.099191</td>\n",
       "      <td>79.281714</td>\n",
       "      <td>50.503006</td>\n",
       "      <td>5.741126</td>\n",
       "      <td>10.408752</td>\n",
       "      <td>3.193904</td>\n",
       "      <td>2.912587</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>50.000000</td>\n",
       "      <td>1.005000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>4.500000</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>3.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>99.750000</td>\n",
       "      <td>42.000000</td>\n",
       "      <td>70.000000</td>\n",
       "      <td>1.010000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0.900000</td>\n",
       "      <td>135.000000</td>\n",
       "      <td>3.800000</td>\n",
       "      <td>10.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>199.500000</td>\n",
       "      <td>55.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>1.020000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>121.000000</td>\n",
       "      <td>42.000000</td>\n",
       "      <td>1.300000</td>\n",
       "      <td>138.000000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>12.650000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>299.250000</td>\n",
       "      <td>64.500000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>1.020000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>163.000000</td>\n",
       "      <td>66.000000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>142.000000</td>\n",
       "      <td>4.900000</td>\n",
       "      <td>15.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>399.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>1.025000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>490.000000</td>\n",
       "      <td>391.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>163.000000</td>\n",
       "      <td>47.000000</td>\n",
       "      <td>17.800000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               id         age          bp          sg          al          su  \\\n",
       "count  400.000000  391.000000  388.000000  353.000000  354.000000  351.000000   \n",
       "mean   199.500000   51.483376   76.469072    1.017408    1.016949    0.450142   \n",
       "std    115.614301   17.169714   13.683637    0.005717    1.352679    1.099191   \n",
       "min      0.000000    2.000000   50.000000    1.005000    0.000000    0.000000   \n",
       "25%     99.750000   42.000000   70.000000    1.010000    0.000000    0.000000   \n",
       "50%    199.500000   55.000000   80.000000    1.020000    0.000000    0.000000   \n",
       "75%    299.250000   64.500000   80.000000    1.020000    2.000000    0.000000   \n",
       "max    399.000000   90.000000  180.000000    1.025000    5.000000    5.000000   \n",
       "\n",
       "              bgr          bu          sc         sod         pot        hemo  \n",
       "count  356.000000  381.000000  383.000000  313.000000  312.000000  348.000000  \n",
       "mean   148.036517   57.425722    3.072454  137.528754    4.627244   12.526437  \n",
       "std     79.281714   50.503006    5.741126   10.408752    3.193904    2.912587  \n",
       "min     22.000000    1.500000    0.400000    4.500000    2.500000    3.100000  \n",
       "25%     99.000000   27.000000    0.900000  135.000000    3.800000   10.300000  \n",
       "50%    121.000000   42.000000    1.300000  138.000000    4.400000   12.650000  \n",
       "75%    163.000000   66.000000    2.800000  142.000000    4.900000   15.000000  \n",
       "max    490.000000  391.000000   76.000000  163.000000   47.000000   17.800000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kd_dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "99695783",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 400 entries, 0 to 399\n",
      "Data columns (total 26 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   id              400 non-null    int64  \n",
      " 1   age             391 non-null    float64\n",
      " 2   bp              388 non-null    float64\n",
      " 3   sg              353 non-null    float64\n",
      " 4   al              354 non-null    float64\n",
      " 5   su              351 non-null    float64\n",
      " 6   rbc             248 non-null    object \n",
      " 7   pc              335 non-null    object \n",
      " 8   pcc             396 non-null    object \n",
      " 9   ba              396 non-null    object \n",
      " 10  bgr             356 non-null    float64\n",
      " 11  bu              381 non-null    float64\n",
      " 12  sc              383 non-null    float64\n",
      " 13  sod             313 non-null    float64\n",
      " 14  pot             312 non-null    float64\n",
      " 15  hemo            348 non-null    float64\n",
      " 16  pcv             330 non-null    object \n",
      " 17  wc              295 non-null    object \n",
      " 18  rc              270 non-null    object \n",
      " 19  htn             398 non-null    object \n",
      " 20  dm              398 non-null    object \n",
      " 21  cad             398 non-null    object \n",
      " 22  appet           399 non-null    object \n",
      " 23  pe              399 non-null    object \n",
      " 24  ane             399 non-null    object \n",
      " 25  classification  400 non-null    object \n",
      "dtypes: float64(11), int64(1), object(14)\n",
      "memory usage: 81.4+ KB\n"
     ]
    }
   ],
   "source": [
    "kd_dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2ece5f32",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns_ret=['sg','al','sc','hemo','pcv','wbcc','rbcc','htn','classification']\n",
    "kd_dataset=kd_dataset.drop([col for col in kd_dataset.columns if not col in columns_ret],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "05bb50a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "kd_dataset=kd_dataset.dropna(axis=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b78e6e2",
   "metadata": {},
   "source": [
    "Non-Numeric Data Transformation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "2c5eeccc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SUBBA REDDY\\AppData\\Local\\Temp\\ipykernel_9392\\2813033204.py:2: DeprecationWarning: Converting `np.inexact` or `np.floating` to a dtype is deprecated. The current result is `float64` which is not strictly correct.\n",
      "  if kd_dataset[column].dtype == npy.number:\n"
     ]
    }
   ],
   "source": [
    "for column in kd_dataset.columns:\n",
    "    if kd_dataset[column].dtype == npy.number:\n",
    "        continue\n",
    "    kd_dataset[column]=LabelEncoder().fit_transform(kd_dataset[column])    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1b439ec1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sg</th>\n",
       "      <th>al</th>\n",
       "      <th>sc</th>\n",
       "      <th>hemo</th>\n",
       "      <th>pcv</th>\n",
       "      <th>htn</th>\n",
       "      <th>classification</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.020</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>15.4</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.020</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>11.3</td>\n",
       "      <td>22</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.010</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.8</td>\n",
       "      <td>9.6</td>\n",
       "      <td>15</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.005</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.8</td>\n",
       "      <td>11.2</td>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.010</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>11.6</td>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      sg   al   sc  hemo  pcv  htn  classification\n",
       "0  1.020  1.0  1.2  15.4   28    1               0\n",
       "1  1.020  4.0  0.8  11.3   22    0               0\n",
       "2  1.010  2.0  1.8   9.6   15    0               0\n",
       "3  1.005  4.0  3.8  11.2   16    1               0\n",
       "4  1.010  2.0  1.4  11.6   19    0               0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "kd_dataset.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a63c29f7",
   "metadata": {},
   "source": [
    "Splitting the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ff2ccc85",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=kd_dataset.drop(['classification'],axis=1)\n",
    "Y=kd_dataset['classification']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0e9823dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        sg   al   sc  hemo  pcv  htn\n",
      "0    1.020  1.0  1.2  15.4   28    1\n",
      "1    1.020  4.0  0.8  11.3   22    0\n",
      "2    1.010  2.0  1.8   9.6   15    0\n",
      "3    1.005  4.0  3.8  11.2   16    1\n",
      "4    1.010  2.0  1.4  11.6   19    0\n",
      "..     ...  ...  ...   ...  ...  ...\n",
      "395  1.020  0.0  0.5  15.7   31    0\n",
      "396  1.025  0.0  1.2  16.5   38    0\n",
      "397  1.020  0.0  0.6  15.8   33    0\n",
      "398  1.025  0.0  1.0  14.2   35    0\n",
      "399  1.025  0.0  1.1  15.8   37    0\n",
      "\n",
      "[287 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "a06cabaf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0      0\n",
      "1      0\n",
      "2      0\n",
      "3      0\n",
      "4      0\n",
      "      ..\n",
      "395    1\n",
      "396    1\n",
      "397    1\n",
      "398    1\n",
      "399    1\n",
      "Name: classification, Length: 287, dtype: int32\n"
     ]
    }
   ],
   "source": [
    "print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c5c5330c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "60d44ea4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(287, 6) (229, 6) (58, 6)\n"
     ]
    }
   ],
   "source": [
    "print(X.shape,X_train.shape,X_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e240c0fe",
   "metadata": {},
   "source": [
    "Training The Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1b64b92e",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "9b28d519",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model1.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e866b6c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_pred=model1.predict(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "bfac6089",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_accry=accuracy_score(X_train_pred,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2841eafe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of Trained Model : 0.9868995633187773\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy of Trained Model :\",X_train_accry)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "207769b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_pred=model1.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "8b72e5bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_accry=accuracy_score(X_test_pred,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9a1d12f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of Trained Model : 0.9137931034482759\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy of Trained Model :\",X_test_accry)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "60b1a64d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n",
      "[1]\n",
      "Kidney disease\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SUBBA REDDY\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "data=(1.020,0.0,0.6,15.8,33,0)\n",
    "data_as_array=npy.asarray(data)\n",
    "data_reshaped=data_as_array.reshape(1,-1)\n",
    "prediction=model1.predict(data_reshaped)\n",
    "print(prediction)\n",
    "print(prediction)\n",
    "if(prediction[0]==0):\n",
    "    print(\"No Kidney disease\")\n",
    "else:\n",
    "    print(\"Kidney disease\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efa2d08c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
