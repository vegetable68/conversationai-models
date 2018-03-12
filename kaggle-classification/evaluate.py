import numpy as np
import pandas as pd
import tensorflow as tf

# Name of the input words feature
WORDS_FEATURE = 'words'

def predict():

    with tf.Session(graph=tf.Graph()) as sess:

        export_dir = 'saved_models/1517848942/'

        model_input = tf.train.Example(
            features=tf.train.Features(
                feature={
                    WORDS_FEATURE: tf.train.Feature(
                        int64_list=tf.train.Int64List(value=feature)
                    )
                }
            )
        )

        model_input = model_input.SerializeToString()
        output_dict = classifier({'inputs': [model_input]})
        scores = output_dict['scores']
        predicted_class = np.argmax(scores)

def load_model(session, model_dir):
    # load saved graph ???
    tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_dir)

    # load ... ???
    classifier = tf.contrib.predictor.from_saved_model(model_dir)

    return classifier

def main():

    with tf.Session(graph=tf.Graph()) as sess:
        classifier = load_model(sess, FLAGS.model_path)

    # load test data
    vocab_processor_path =

    data = wikidata.WikiData(
        FLAGS.train_data, FLAGS.y_class, seed=DATA_SEED, train_percent=TRAIN_PERCENT,
        max_document_length=MAX_DOCUMENT_LENGTH, model_dir=FLAGS.saved_model_dir)

    data = wikidata.WikiData(
      FLAGS.data_path, FLAGS.y_class, max_document_length=MAX_DOCUMENT_LENGTH)

    import pdb; pdb.set_trace()

    # evaluate model on data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', help='Path to data to evaluate on.', type='str')
    parser.add_argument('--model_dir', help='Path to directory with TF model.', type='str')
    parser.add_argument('--y_class', type=str, default="toxic")


    FLAGS, unparsed = parser.parse_known_args()

    main()
