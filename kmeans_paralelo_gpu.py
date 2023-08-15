
import cudf
import random
import tqdm
import cupy as cp

import time


#inicia centroides aleatoriamente a partir de valores do dataframe
def init_centroides(n_center, df_data, df_ctr):
    for i in range(0, n_center):
        index_1 = random.randint(0,len(df_data))
        index_2 = random.randint(0,len(df_data))
        for column in df_data.columns:
            valor_1 =  df_data.loc[index_1,column]
            valor_2 =  df_data.loc[index_2,column]
            if df_data[column].dtype != 'object': 
                v = (valor_1 + valor_2)/2 #operação vetorial para fazer media dos valores aleotorios que serão os valores de 1 centroide
            else:
                v = valor_1
            df_ctr.loc[i, column] = v
    return df_ctr


def norm(df):
    #normaliza a coluna aplicando a formula de normalização de maneira vetorial por toda coluna 
    for column in df.columns:
        if df[column].dtype != 'object':
            max_value = df[column].max()
            min_value = df[column].min()
            df[column] = (df[column] - min_value) / (max_value - min_value)
    return df     




def conv_col(df, indices, type_n):
    #casting de dados de uma coluna
    colunas = df.columns[indices]
    for coluna in colunas:
        df[coluna] = df[coluna].astype(type_n)
        
    return df


def update_centroide(indices,df_ctr,df_data):
    #uso da função pronta do cudf para fazer media dos valores 
    df_ctr = df_data.groupby('centroide').mean().sort_index()
    return df_ctr


def get_euclidian_distance_new(indices, df_ctr, df_data):
    ctr = df_ctr.iloc[:, indices].astype('float64')
    dt = df_data.iloc[:, indices].astype('float64')
    dist_array = []
    #a função euclidiana aqui foi implementada como uma serie de operações vetoriais
    for i in range(0,len(ctr)):
        linha = ctr.iloc[i,:] #valores da linha
        squared_diff = (dt-linha)**2 #operação vetorial para realizar a diferença ao quadrado de cada valor da linha paralelamnete
        sum_squared_diff = squared_diff.T.sum() #soma os valores da transposta das diferenças, assim temos um veotr  que é a soma das distancias da linha dos dados e centroides
        dist = sum_squared_diff ** 0.5 #
        dist_array.append(dist) #adiciona esse vetor de distancia entre uma linha e cada centroide em um array
        #esse pr

    df = cudf.DataFrame(dist_array).T #transposta do array de distancias entre os centroides e as linhas
    array_result = df.to_cupy()

    min_columns_index = cp.argmin(array_result, axis=1) #pega o centroide de menor distancia dentro do array de distancias 
    df_data['centroide'] = min_columns_index #atualiza os centroides


ini = time.time()
n_center = 4
n_iters = 10
indices = [0,1,2,3,4,5,6,7,8]
df_data = cudf.read_csv('dataset/housing.csv')
df_data.insert(10,'centroide','x')
df_data = df_data.dropna()
df_ctr = cudf.DataFrame(index=range(n_center), columns=df_data.columns)

df_data = norm(df_data)
df_ctr = init_centroides(n_center, df_data, df_ctr)

print(f"Running kmeans for {n_iters} iterations")
for i in tqdm.tqdm(range(0,n_iters)):
    get_euclidian_distance_new(indices, df_ctr, df_data)
    df_ctr = update_centroide(indices,df_ctr,df_data)


fim = time.time()
print(f'Duration: {fim-ini}')
