import theano
import lasagne
import theano.tensor as T

def compile_theano_expr(def_net_arch, update_method):
    X_batch = T.tensor4('x')
    y_batch = T.ivector('y')
    learning_rate = T.scalar('lr')
    # momentum = T.scalar('momt')
    net_arch = def_net_arch(X_batch)

    output = lasagne.layers.get_output(net_arch['softmax_out'], X_batch, deterministic=False)

    loss = lasagne.objectives.categorical_crossentropy(output, y_batch)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(net_arch['softmax_out'], trainable=True)
    updates = update_method(loss, params, learning_rate)

    output_test = lasagne.layers.get_output(net_arch['softmax_out'], X_batch, deterministic=True)

    loss_test = lasagne.objectives.categorical_crossentropy(output, y_batch)
    loss_test = loss_test.mean()
    acc_test = T.mean(T.eq(T.argmax(output_test, axis=1), y_batch), dtype = theano.config.floatX)

    train_fn = theano.function(
        [X_batch, y_batch, learning_rate], loss, 
        updates = updates,
        allow_input_downcast=True
        )

    test_fn = theano.function( 
        [X_batch, y_batch], [loss_test, acc_test],
        allow_input_downcast=True
        )

    return dict(
        train=train_fn,
        test=test_fn,
        net_arch = net_arch,
        )