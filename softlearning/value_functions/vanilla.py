from softlearning.models.feedforward import feedforward_model, cnn_model, head_model


def create_feedforward_Q_function(observation_shape,
                                  action_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_Q',
                                  shared_inputs=None,
                                  shared_outputs=None,
                                  **kwargs):
    input_shapes = (observation_shape, action_shape)
    preprocessors = (observation_preprocessor, None)
    # return feedforward_model(
    #     input_shapes,
    #     *args,
    #     output_size=1,
    #     preprocessors=preprocessors,
    #     name=name,
    #     **kwargs)
    return head_model(
            input_shapes = [action_shape],
            output_size = 1,
            shared_inputs = shared_inputs,
            shared_outputs = shared_outputs
        )


def create_feedforward_V_function(observation_shape,
                                  *args,
                                  observation_preprocessor=None,
                                  name='feedforward_V',
                                  **kwargs):
    input_shapes = (observation_shape, )
    preprocessors = (observation_preprocessor, None)
    return feedforward_model(
        input_shapes,
        *args,
        output_size=1,
        preprocessors=preprocessors,
        **kwargs)
