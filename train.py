
import torch
from torch.autograd import Variable


device = torch.device("cpu") # device

from data import DataLoader
import model
from plot import plot_result

def main():
    data_loader=DataLoader()
    df,X_train_tensors_final, y_train_tensors, X_test_tensors_final, y_test_tensors,ss,mm = data_loader.load_data()

    dataY_plot,data_predict ,loss_arr = result(df,X_train_tensors_final, y_train_tensors,ss,mm)

    plot_result(dataY_plot,data_predict ,loss_arr)
def train_model(X_train_tensors_final,y_train_tensors):
    num_epochs= 1000#1000 epochs
    learning_rate=0.000003#0.001 lr

    input_size=5 #number of features
    hidden_size=2#number of features in hidden state
    num_layers=1#number of stacked lstm layers
    num_classes=1#number of output classes

    lstm1 = model.LSTM1( num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

    loss_function = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer


    loss_arr=[]
    for epoch in range(num_epochs):
        outputs = lstm1.forward(X_train_tensors_final.to(device)) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_function(outputs, y_train_tensors.to(device))
        loss_arr.append(loss)
        loss.backward() #calculates the loss of the loss function
        optimizer.step() #improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    return lstm1,loss_arr

def result(df,X_train_tensors_final,y_train_tensors,ss,mm):

    model,loss_arr = train_model(X_train_tensors_final,y_train_tensors)

    df_X_ss = ss.transform(df.drop(columns='Volume'))
    df_y_mm = mm.transform(df.iloc[:, 5:6])

    df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
    df_y_mm = Variable(torch.Tensor(df_y_mm))

    #reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
    train_predict = model(df_X_ss.to(device))#forward pass
    data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict) #reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)

    return dataY_plot,data_predict ,torch.tensor(loss_arr)
if __name__ == '__main__':
    main()