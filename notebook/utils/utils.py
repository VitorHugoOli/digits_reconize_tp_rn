import tensorflow as tf

def _fit_eval(model, x, y, x_valid=None, y_valid=None, epochs=10, batch_size=128):
    
    model.fit(x, y,
          epochs=10,
          verbose=1,
          validation_data=(x_valid, y_valid))
          
    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])