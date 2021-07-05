import pandas as pd
import glob

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
    shelf_path = [path for path in filepaths if 'attributes' not in path][0]
    df = pd.read_json(shelf_path)
    
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