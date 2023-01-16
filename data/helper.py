import pandas as pd
from pathlib import Path

def guardarPuntuacion(punto,name='datos.csv'):
    puntoL=[punto]
    data = {'Puntos': puntoL}
    data_path = ('./data/'+name)
    #print(data_path)
    if Path(data_path ).exists():
        #print('existe')
        df = pd.read_csv(data_path )
        #print(data)
        df2 = pd.DataFrame(data, columns=['Puntos'])
        #print(df2)
        # Apilar los __DataFrames__ uno encima del otro
        stack = pd.concat([df, df2], axis=0)
        stack.to_csv(data_path , index=False)
    else:
        #print('no existe')
        df = pd.DataFrame(data, columns=['Puntos'])
        df.to_csv(data_path , index=False)