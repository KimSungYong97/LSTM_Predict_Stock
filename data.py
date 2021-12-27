
import pandas_datareader.data as pdr
import datetime
import torch
from torch.autograd import Variable
class DataLoader:
    def __init__(self):
        start = (2000, 1, 1)  # 2020년 01년 01월~
        start = datetime.datetime(*start)
        end = datetime.date.today()  # ~현재
        self.df = pdr.DataReader('005930.KS', 'yahoo', start, end)  # yahoo 에서 삼성 전자 불러오기

        '''
            open 시가
            high 고가
            low 저가 
            close 종가 
            volume 거래량 
            Adj Close 주식의 분할, 배당, 배분 등을 고려해 조정한 종가
        '''

    def Scale_data(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        X = self.df.drop(columns='Volume')
        y = self.df.iloc[:, 5:6]
        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
        X_ss = self.ss.fit_transform(X)
        y_mm = self.mm.fit_transform(y)
        # Train Data
        self.X_train = X_ss[:4500, :]
        self.X_test = X_ss[4500:, :]
        # Test Data
        self.y_train = y_mm[:4500, :]
        self.y_test = y_mm[4500:, :]

        print("Training Shape", self.X_train.shape, self.y_train.shape)
        print("Testing Shape", self.X_test.shape, self.y_test.shape)


    def load_data(self):
        self.Scale_data()
        X_train_tensors = Variable(torch.Tensor(self.X_train))
        X_test_tensors = Variable(torch.Tensor(self.X_test))
        y_train_tensors = Variable(torch.Tensor(self.y_train))
        y_test_tensors = Variable(torch.Tensor(self.y_test))
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
        X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
        print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

        return self.df,X_train_tensors_final, y_train_tensors, X_test_tensors_final, y_test_tensors,self.ss,self.mm

