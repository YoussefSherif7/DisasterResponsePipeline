import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
        inputs: messages filepath, categories filepath
            
        output: messages and categories as merged data frame 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)

    return df


def clean_data(df):
    """
        inputs: data frame
            
        output: cleaned data frame with correct category columns and 0/1 category values  
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # rename the columns of `categories`
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Handling case where values are (2)
    categories = categories.replace(2, 0)
    
    #Replace categories column in df with new category columns. 
    # drop the original categories column from `df`
    df=df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
def save_data(df, database_filename):
    """
        Saves the clean dataframe to a SQL database
        inputs: data frame
            
    """
    #Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Responses', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()