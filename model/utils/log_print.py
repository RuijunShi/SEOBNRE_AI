def log_str(epoch, num_epoch, train_loss, ii, test_loss, lr, end_time, start_time):
    str = f"Epoch [{epoch+1}/{num_epoch}]|Loss:{(train_loss/(ii+1))}|test loss:{(test_loss)}|lr:{lr}|Time:{end_time-start_time:.2f}s|"

    return str
