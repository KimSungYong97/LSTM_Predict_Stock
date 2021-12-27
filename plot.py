import matplotlib.pyplot as plt

def plot_result(dataY_plot,data_predict,loss_arr):
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=4500, c='r', linestyle='--') #size of the training set

    plt.plot(dataY_plot, label='Actuall Data') #actual
    plt.plot(data_predict, label='Predicted Data') #pred
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()
    plt.savefig('Prediction.png')

    plt.figure(figsize=(10,6)) #plotting
    plt.plot(loss_arr) #loss plot
    plt.title('Time-Series Loss')
    plt.show()
    plt.savefig('loss.png')