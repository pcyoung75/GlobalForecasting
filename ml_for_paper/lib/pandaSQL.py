import pandas.io.sql as psql
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from patsy.highlevel import dmatrices

class PandaSQL():
    def __init__(self):
        pass

    def getDF(self, db, table, where=None, columns=None):
        ##################################
        # 1. Access MySQL
        user = "root"
        password = "jesus"
        ip = "127.0.0.1"

        conn = create_engine('mysql://' + user + ':' + password + '@' + ip + '/' + db + '?charset=utf8')

        query = None

        if columns is not None:
            columns = ', '.join(columns)

        if where is None:
            if columns is None:
                query = ("SELECT * FROM {}.{} ".format(db, table))
            else:
                query = ("SELECT {} FROM {}.{} ".format(columns, db, table))
        else:
            if columns is None:
                query = ("SELECT * FROM {}.{} {}".format(db, table, where))
            else:
                query = ("SELECT {} FROM {}.{} {}".format(columns, db, table, where))

        print('SQL: ' + query)
        df = pd.read_sql(query, con=conn)

        return df

    def getXsYData(self, df, yIndex, xlist):
        modelelements = ' + '.join(xlist)
        formula = yIndex + ' ~ ' + modelelements
        # print(formula)

        y, X = dmatrices(formula, data=df, return_type='dataframe')
        X = X.drop('Intercept', 1)

        return y, X

    def getSelectedData(self, df, yIndex, xlist):
        modelelements = ' + '.join(xlist)
        formula = yIndex + ' ~ ' + modelelements
        # print(formula)

        y, x = dmatrices(formula, data=df, return_type='dataframe')
        x = x.drop('Intercept', 1)

        return pd.concat([x, y], axis=1)
