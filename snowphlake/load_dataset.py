def load_dataset(name=None):
    
    if name==None:
        name = 'splData'
    import pandas as pd
    import requests
    import io
    
    if name == 'EBMData':
        urlname = 'https://raw.githubusercontent.com/88vikram/pyebm/master/resources/Data_7.csv'
    elif name == 'splData':
        urlname = 'https://raw.githubusercontent.com/snowphlake-dpm/snowphlake/main/resources/toy_dataset.csv'
    s=requests.get(urlname).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    return df