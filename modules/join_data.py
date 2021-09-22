import pandas as pd
import glob
from hashlib import sha256


# Contains code to compile the shelf data file with attribute data files
# a folder must contain only linking data

def get_filepaths(
    directory,
    wildcard=None,
):
    """Get filepaths for all files in a directory, can put a wildcard keep only specifics
    'C:\\Users\\Administrator\\expenditure_proxies\\sample_data\\*'  
    """
    paths = glob.glob(directory)
    if wildcard != None:
        paths = [path for path in paths if wildcard in path]
    
    return paths


def __read_shelf(
    filepaths,
):
    """Reads shelf data given the filepaths
    """
    shelf_paths = [path for path in filepaths if 'attributes' not in path]
    df = pd.read_json(shelf_paths[0])
    for file in shelf_paths[1:]:
        df = pd.concat([df, pd.read_json(file)]).reset_index().drop('index', axis=1)
    
    return df


def __clean_remotekey(
    df,
):
    """Removes remotekey slash and converts to int
    """
    df['remotekey'] = df['remotekey'].str[0][0].replace('/','')
    df['remotekey'] = df['remotekey'].astype('int64')
    
    return df


def __merge_attributes(
    filepaths,
):
    """Merges attributes files together
    """
    attribute_paths = [path for path in filepaths if 'attributes' in path]
    attribute_df = pd.read_json(attribute_paths[0], orient='index')
    attribute_df = __clean_remotekey(attribute_df)
    
    for path in attribute_paths[1:]:
        temp_df = pd.read_json(path, orient='index')
        temp_df = __clean_remotekey(temp_df)
        attribute_df = pd.concat([attribute_df, temp_df])
    
    attribute_df.reset_index(inplace=True)
    attribute_df.drop('index', axis=1, inplace=True)
    
    return attribute_df


def __join_2shelf(
    attribute_df,
    filepaths,
):
    """Joins attribute data to shelf data
    """
    shelf_df = __read_shelf(filepaths)
    df = pd.merge(shelf_df, attribute_df, on='remotekey', how='outer')
    
    return df


def compile_data(
    directory_path,
    category_name,
):
    """Master function which will compile the shelf data with its attribute files
       into one dataframe and then write back out in current directory.
    """
    filepaths = get_filepaths(directory_path)
    attribute_df = __merge_attributes(filepaths)
    df = __join_2shelf(attribute_df, filepaths)
    df.to_json(directory_path[:-1]+category_name+'.json')
    

def letters_to_numbers(
    string,
):
    """Converts all letters in a string to a number
    """
    new_string = ''
    for char in string:
        if char.isdigit():
            new_string += str(char)
        else:
            new_string += str(ord(char) - 96)
            
    return new_string
        
    
def encode_retailer_code(
    df,
):
    """Encodes retailer code
    """
    df['code'] = df['remotekey'].apply(lambda x: (sha256(repr(x).encode('utf-8')).hexdigest()))
    df['code'] = df['code'].apply(lambda x: letters_to_numbers(x))
    df['code'] = df['code'].astype('str')
    
    return df
    

def join_y(
    df,
    y,
):
    """Encodes scraped retail codes and joins the y data
    """
    df = encode_retailer_code(df)
    sales_df = pd.read_csv(y)
    sales_df.drop('Unnamed: 0', axis=1, inplace=True)
    sales_df['code'] = sales_df['code'].astype('str')
    df = df.merge(sales_df, on='code')
    
    return df
    
    